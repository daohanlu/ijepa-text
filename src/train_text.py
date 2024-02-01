# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import decorator
from transformers import DataCollatorForLanguageModeling

from src.masks.my_collators import MyDataCollatorForT5MLM

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.c4 import make_c4, get_tokenizer
from src.masks.my_collators import compute_input_and_target_lengths

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_batch(batch, device):
    masked_input_ids = batch['masked_input_ids'].to(device, non_blocking=True)
    unmasked_input_ids = batch['unmasked_input_ids'].to(device, non_blocking=True)
    target_ids = batch['target_ids'].to(device, non_blocking=True)
    return masked_input_ids, unmasked_input_ids, target_ids


def load_batch_2(batch, device):
    unmasked_input_ids = batch['input_ids'].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True)
    noise_mask = batch['noise_mask'].to(device, non_blocking=True)
    return unmasked_input_ids, labels, noise_mask


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    input_length = args['data']['input_length']
    mlm_probability = args['data']['mlm_probability']
    mean_noise_span_length = args['data']['mean_noise_span_length']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    iterations_per_epoch = args['optimization']['iterations_per_epoch']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    os.makedirs(folder, exist_ok=True)
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    tokenizer = get_tokenizer()
    # tokenizer.mask_token = 32000
    encoder, predictor = init_model(
        device=device,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        n_positions=input_length
    )
    target_encoder = copy.deepcopy(encoder)

    before_mask_input_length, target_length = compute_input_and_target_lengths(
        inputs_length=input_length,
        noise_density=mlm_probability,
        mean_noise_span_length=mean_noise_span_length,
    )
    logger.info(f'input length: {input_length}; before_mask_input_length: {before_mask_input_length}, '
                f'target_length: {target_length}.')
    logger.info(f'device: {device}, encoder device: {next(encoder.parameters()).device},'
                f' predictor device: {next(predictor.parameters()).device}')

    text_collator = MyDataCollatorForT5MLM(tokenizer=tokenizer,
                                           noise_density=mlm_probability,
                                           mean_noise_span_length=mean_noise_span_length,
                                           input_length=input_length,
                                           target_length=target_length,
                                           pad_token_id=tokenizer.pad_token_id,
                                           decoder_start_token_id=tokenizer.pad_token_id, )

    # -- init data-loaders/samplers
    dataset, _, dataloader = make_c4(
        batch_size=batch_size,
        collator=text_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        before_mask_input_length=before_mask_input_length,
        tokenizer=tokenizer
    )
    ipe = iterations_per_epoch

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    # --- DistributedDataParallel ---
    # encoder = DistributedDataParallel(encoder, static_graph=True)
    # predictor = DistributedDataParallel(predictor, static_graph=True)
    # target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
                          for i in range(int(ipe * num_epochs * ipe_scale) + 1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    itr = 0
    dataloader_iterator = iter(dataloader)
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        # unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        # for itr, batch in enumerate(dataloader):
        current_ema_momentum = 0
        for _ in range(ipe):
            batch = next(dataloader_iterator)
            # masked_input_ids, unmasked_input_ids, target_ids = load_batch(batch, device)
            # masked_indices: A bool Tensor (B x sequence_length) set to True when the token is masked out.
            unmasked_input_ids, labels, noise_mask = load_batch_2(batch, device)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(unmasked_input_ids)
                        h_before_norm = h.detach().clone()
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        # -- create targets (masked regions of h)
                        B, N, D = h.shape
                        h = h[noise_mask].reshape(B, -1, D)
                        # h = repeat_interleave_batch(h, B, repeat=len(noise_mask))
                        return h

                def forward_context():
                    # encoder mask: not masked_indices;
                    z = encoder(unmasked_input_ids, ~noise_mask)
                    # encoder mask: not masked_indices; decoder mask: masked_indices
                    z = predictor(z, ~noise_mask, noise_mask)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with (torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16)):
                    h = forward_target()
                    z = forward_context()
                    # shape sanity checks: number of tokens predicted should be target_length - 1
                    B, N, D = h.shape
                    assert B == batch_size and N == target_length - 1, \
                        f'h shape should be {(batch_size, target_length - 1, "*")}, but got {h.shape}!'
                    B, N, D = z.shape
                    assert B == batch_size and N == target_length - 1, \
                        f'z shape should be {(batch_size, target_length - 1, "*")}, but got {z.shape}!'
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

                return float(loss), _new_lr, _new_wd, grad_stats, m

            (loss, _new_lr, _new_wd, grad_stats, current_ema_momentum), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024. ** 2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()
            itr += 1
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info(f'avg. loss: {loss_meter.avg:.3f}. Current EMA weight: {current_ema_momentum:.6f}')
        save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
