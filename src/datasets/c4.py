import pdb
from logging import getLogger

import datasets
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = getLogger()


def get_tokenizer(model_name='google/t5-v1_1-base'):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        legacy=False
    )
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def tokenize_function(examples, tokenizer, in_length):
    tokenizer_out = tokenizer(
        text=examples["text"],
        padding='longest',
        max_length=in_length,
        truncation=True,
        return_attention_mask=False,
        return_tensors='pt',
    )
    result = {"input_ids": tokenizer_out["input_ids"]}
    return result


def make_c4(
        batch_size,
        before_mask_input_length: int,
        collator=None,
        pin_mem=True,
        num_workers=16,
        world_size=1,
        rank=0,
        training=True,
        drop_last=True,
        tokenizer: PreTrainedTokenizer = None,
):
    split = 'train' if training else 'validation'
    logger.info('Loading C4 dataset with streaming=True')
    dataset = datasets.load_dataset('allenai/c4', 'en', split=split, streaming=True)
    dataset = dataset.remove_columns(
        ['timestamp', 'url']
    )
    assert dataset.n_shards >= 1024, \
        ("We want to have >=1024 shards for efficient processing with num_workers in PyTorch dataloader. "
         "The number of shards is: " + str(dataset.n_shards))
    # ---Tokenize and shuffle---
    logger.info(f"tokenizer is {tokenizer.__class__} and tokenizer.pad_token_id is {tokenizer.pad_token_id}.")

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={
            'tokenizer': tokenizer,
            'in_length': before_mask_input_length,
        },
        remove_columns=['text'],
    )  # this removes column 'text' and adds column 'input_ids' corresponding to converted token IDs.
    dataset = dataset.shuffle(buffer_size=10_000, seed=42)
    # dataset = dataset.with_format("torch")
    # dist_sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset=dataset,
    #     num_replicas=world_size,
    #     rank=rank)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     collate_fn=collator,
    #     sampler=dist_sampler,
    #     batch_size=batch_size,
    #     drop_last=drop_last,
    #     pin_memory=pin_mem,
    #     num_workers=num_workers,
    #     persistent_workers=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('C4 dataset created')
    return dataset, None, data_loader


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """
    Modified to compute lengths assume we replace each length-N masked span with N mask tokens
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """
    targets_length = int(round(inputs_length * noise_density))
    # subtract 1 to accommodate the EOS token.
    return inputs_length - 1, targets_length - 1