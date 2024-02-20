# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import decorator
import matplotlib.pyplot as plt
from transformers import DataCollatorForLanguageModeling

from src.masks.my_collators import MyDataCollatorForLanguageModeling
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
import pdb

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


def load_batch_t5_mlm(batch, device):
    uncorrupted_input_ids = batch['input_ids'].to(device, non_blocking=True)
    noise_mask = batch['noise_mask'].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True)  # noisy input_ids via BERT-style MLM
    return uncorrupted_input_ids, noise_mask, labels


def load_batch_bert_mlm(batch, device):
    uncorrupted_input_ids = batch['uncorrupted_input_ids'].to(device, non_blocking=True)
    noise_mask = batch['noise_mask'].to(device, non_blocking=True)
    input_ids = batch['input_ids'].to(device, non_blocking=True)  # noisy input_ids via BERT-style MLM
    labels = batch['labels'].to(device, non_blocking=True)  # noisy input_ids via BERT-style MLM
    return uncorrupted_input_ids, noise_mask, input_ids, labels


def make_plot(title, xlabel, ylabel, ys, save_path, scatter=False):
    # Create a plot of the singular values
    xs = np.arange(1, len(ys) + 1)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    if scatter:
        ax.scatter(xs, ys, marker='.')
    else:
        ax.plot(xs, ys, marker='.')
    ax.set_title(title)
    # ax.set_xticks(xs)
    ax.set_xlim(1 - 1, len(ys) + 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def off_diagonal(x):
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def check_stats(buffer, h, z, is_random_init, experiment_name):
    """Checks for mode collapse in terms of features
    h: target features. B x N x D
    z: predicted features. B x N x D"""
    h_buffer = buffer['h']
    z_buffer = buffer['z']
    trial_num = buffer['trial_num']
    h_buffer.append(h.cpu())
    z_buffer.append(z.cpu())
    assert len(h_buffer) == len(z_buffer)
    num_tokens = 0 if len(h_buffer) == 0 else len(h_buffer) * h_buffer[0].shape[0] * h_buffer[0].shape[1]
    if num_tokens >= 1024 * 128:
        logger.info(f'VIC debug: collected features from {num_tokens} tokens.')
        h = torch.concatenate(h_buffer, dim=0)  # concat into a large batch
        z = torch.concatenate(z_buffer, dim=0)  # concat into a large batch
        assert h.shape == z.shape and h.shape[0] * h.shape[1] == num_tokens
        B, N, D = h.shape
        h1 = h.reshape(-1, D)
        var = h1.var(dim=0)
        cov = h1.T.cov()
        mu = h1.mean(dim=0)
        log_det = torch.logdet(cov)
        U, S, V = torch.svd(cov)
        cholesky_mesg = "Cholesky success"
        try:
            L = torch.linalg.cholesky(cov)
        except torch._C._LinAlgError:
            cholesky_mesg = "Cholesky failed"
        model_name = 'Random init. model. ' if is_random_init else 'Pre-trained model. '
        save_prefix = 'random_init' if is_random_init else 'pretrained'
        if is_random_init:
            experiment_name = 'random_init'
        svg_folder = os.path.join('plots', experiment_name)
        png_folder = os.path.join('plots_png', experiment_name)
        print('experiment name: ', experiment_name)
        os.makedirs(svg_folder, exist_ok=True)
        os.makedirs(png_folder, exist_ok=True)
        make_plot(f'{model_name} Singular values. Condition number={S.max() / S.min():.3e}. Log det={log_det}. {cholesky_mesg}',
                                  'Sorted Rank', 'Value', S.numpy(), f'{svg_folder}/{save_prefix}__singular_values__trial_{trial_num}.svg')
        make_plot(f'{model_name} Singular values. Condition number={S.max() / S.min():.3e}. Log det={log_det}. {cholesky_mesg}',
                                  'Sorted Rank', 'Value', S.numpy(), f'{png_folder}/{save_prefix}__singular_values__trial_{trial_num}.png')
        make_plot(f'{model_name} Channel-wise Variance. max={var.max():.3e}, min={var.min():.3e}',
                                  'Channel', 'Value', var.numpy(), f'{svg_folder}/{save_prefix}__channel_wise_variance__trial_{trial_num}.svg', scatter=True)
        make_plot(f'{model_name} Channel-wise Variance. max={var.max():.3e}, min={var.min():.3e}',
                                  'Channel', 'Value', var.numpy(), f'{png_folder}/{save_prefix}__channel_wise_variance__trial_{trial_num}.png', scatter=True)
        buffer['trial_num'] += 1
        buffer['h'] = []
        buffer['z'] = []
        if buffer['trial_num'] == 10:
            quit()
        # pdb.set_trace()
    else:
        pass


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    debug_vic = bool(args['meta'].get('debug_vic', False))
    debug_vic_losses = bool(args['meta'].get('debug_vic_losses', False))
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    pred_last_layer_norm = args['meta']['pred_last_layer_norm']
    learnable_pos_embeds = args['meta'].get('learnable_pos_embeds', False)
    is_generative = bool(args['meta'].get('is_generative', False))
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
    use_bert_mlm = args['data'].get('use_bert_mlm', False)
    mlm_probability = args['data']['mlm_probability']
    mean_noise_span_length = args['data'].get('mean_noise_span_length', None)
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
    clip_grad_norm = args['optimization']['clip_grad_norm']
    vicreg_coeff = float(args['optimization'].get('vicreg_coeff', 0.0))
    vicreg_coeff_var = float(args['optimization'].get('vicreg_coeff_var', 0.0))
    vicreg_coeff_cov = float(args['optimization'].get('vicreg_coeff_cov', 0.0))
    if vicreg_coeff > 0:
        assert vicreg_coeff_var == vicreg_coeff_cov == 0, \
            'cannot set combined vicreg coefficient when var and cov coefficients are specified.'
        vicreg_coeff_var = vicreg_coeff * 1.0
        vicreg_coeff_cov = vicreg_coeff * 0.04  # default 1/25 scaling for cov reg

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    if not (debug_vic or debug_vic_losses):
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
    if debug_vic:
        log_file = os.path.join(folder, f'{tag}_r{rank}_debug_vic.csv')
    elif debug_vic_losses:
        log_file = os.path.join(folder, f'{tag}_r{rank}_debug_vic_losses.csv')
    else:
        log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        # only add folder to r_file if it is not a complete path
        if r_file is None or (r_file == os.path.basename(r_file) and not os.path.exists(r_file)):
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        else:
            load_path = r_file

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'VIC-var'),
                           ('%.5f', 'VIC-cov'),
                           ('%d', 'time (ms)'))

    # -- init model
    tokenizer = get_tokenizer()
    # tokenizer.mask_token = 32000
    encoder, predictor = init_model(
        device=device,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        predictor_last_layer_norm=pred_last_layer_norm,
        learnable_pos_embeds=learnable_pos_embeds,
        is_generative=is_generative,
        vocab_size=tokenizer.vocab_size if is_generative else None,
        model_name=model_name,
        n_positions=input_length
    )
    if bool(pred_last_layer_norm) == False:
        assert isinstance(predictor.predictor_norm, torch.nn.Identity)
    else:
        assert isinstance(predictor.predictor_norm, torch.nn.LayerNorm)
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

    if use_bert_mlm:
        text_collator = MyDataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                          mlm_probability=mlm_probability,
                                                          input_length=input_length,
                                                          pad_token_id=tokenizer.pad_token_id,
                                                          decoder_start_token_id=tokenizer.pad_token_id,
                                                          mask_token_id=tokenizer.vocab_size - 100,)
    else:
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
    if debug_vic_losses:
        ipe /= 50  # when debugging vic losses, run fewer iters per epoch

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
            if (epoch) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch}'))

    # -- TRAINING LOOP
    debug_buffer = {'h': [], 'z': [], 'trial_num': 0}
    itr = 0
    dataloader_iterator = iter(dataloader)
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        if debug_vic_losses:
            # load saved model at each epoch
            load_model_epoch = save_path.format(epoch=f'{epoch}')
            encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
                device=device,
                r_path=load_model_epoch,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler)

        # -- update distributed-data-loader epoch
        # unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        var_meter = AverageMeter()
        cov_meter = AverageMeter()
        time_meter = AverageMeter()

        # for itr, batch in enumerate(dataloader):
        current_ema_momentum = 0
        for _ in range(ipe):
            batch = next(dataloader_iterator)
            # masked_indices: A bool Tensor (B x sequence_length) set to True when the token is masked out.
            if use_bert_mlm:
                uncorrupted_input_ids, noise_mask, input_ids, labels = load_batch_bert_mlm(batch, device)
            else:
                uncorrupted_input_ids, noise_mask, labels = load_batch_t5_mlm(batch, device)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(uncorrupted_input_ids)
                        h_before_norm = h.detach().clone()
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        # -- create targets (masked regions of h)
                        B, N, D = h.shape
                        h = h[noise_mask].reshape(B, -1, D)
                        # h = repeat_interleave_batch(h, B, repeat=len(noise_mask))
                        return h

                def forward_context():
                    # encoder mask: not masked_indices;
                    if use_bert_mlm:
                        z = encoder(input_ids)
                        # encoder features should be fully visible since to make use of BERT-style masking
                        z = predictor(z, torch.ones_like(noise_mask), noise_mask)
                    else:
                        z = encoder(uncorrupted_input_ids, ~noise_mask)
                        z = predictor(z, ~noise_mask, noise_mask)
                    return z

                def loss_fn(z, h):
                    if not is_generative:
                        loss = F.smooth_l1_loss(z, h)  # feature-space L1 loss
                    else:
                        loss = F.cross_entropy(z.view(-1, z.size(-1)), h.view(-1), ignore_index=-100)
                    if vicreg_coeff_var > 0 or vicreg_coeff_cov > 0:
                        encoder_features = z.mean(dim=1)  # -> (batch_size, hidden_size)
                        std_x = torch.sqrt(encoder_features.var(dim=0) + 0.0001)  # var over batch dimension
                        cov_x = (encoder_features.T @ encoder_features) / (batch_size - 1)  # cov over batch dimension
                        std_loss = vicreg_coeff_var * (torch.mean(F.relu(1 - std_x)) / 2)
                        cov_loss = vicreg_coeff_cov * (off_diagonal(cov_x).pow_(2).sum().div(z.shape[-1]))  # divide by hidden states dim
                        # see https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py defaults
                        loss += std_loss + cov_loss
                    else:
                        std_loss = 0
                        cov_loss = 0
                    loss = AllReduce.apply(loss)
                    return loss, std_loss, cov_loss

                # Step 1. Forward
                with (torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16)):
                    if not is_generative:
                        h = forward_target()
                    else:
                        # truncate the eos token at the end that is never masked or predicted
                        h = labels[:, :-1].contiguous()
                    z = forward_context()
                    # shape sanity checks: number of tokens predicted should be target_length - 1
                    assert h.shape[0] == batch_size and h.shape[1] == target_length - 1, \
                        f'h shape should be {(batch_size, target_length - 1, "*")}, but got {h.shape}!'
                    assert z.shape[0] == batch_size and z.shape[1] == target_length - 1, \
                        f'z shape should be {(batch_size, target_length - 1, "*")}, but got {z.shape}!'
                    loss, std_loss, cov_loss = loss_fn(z, h)

                if debug_vic:
                    if not load_model:
                        experiment_name = None
                    else:
                        experiment_name = os.path.basename(folder.rstrip("/"))
                    check_stats(debug_buffer, h.detach(), z.detach(), not load_model, experiment_name)

                #  Step 2. Backward & step
                if not (debug_vic or debug_vic_losses):
                    if use_bfloat16:
                        scaler.scale(loss).backward()
                        if clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_norm, norm_type=2.0)
                            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad_norm, norm_type=2.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_norm, norm_type=2.0)
                            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad_norm, norm_type=2.0)
                        optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

                return float(loss), float(std_loss), float(cov_loss), _new_lr, _new_wd, grad_stats, m

            (loss, std_loss, cov_loss, _new_lr, _new_wd, grad_stats, current_ema_momentum), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            var_meter.update(std_loss)
            cov_meter.update(cov_loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, var_meter.val, cov_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   var_meter.avg,
                                   cov_meter.avg,
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
        if not debug_vic:
            save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
