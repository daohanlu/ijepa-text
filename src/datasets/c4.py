import pdb
from logging import getLogger

import datasets
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np

logger = getLogger()


def get_tokenizer(model_name='google/t5-v1_1-base') -> PreTrainedTokenizerBase:
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
        return_attention_mask=False,
    )
    input_ids = tokenizer_out["input_ids"]
    concatenated_ids = np.concatenate(input_ids)
    total_length = concatenated_ids.shape[0]
    total_length = (total_length // in_length) * in_length
    concatenated_ids = concatenated_ids[:total_length].reshape(-1, in_length)
    result = {"input_ids": torch.from_numpy(concatenated_ids)}
    # tokenizer_out = tokenizer(
    #     text=examples["text"],
    #     padding='longest',
    #     max_length=in_length,
    #     truncation=True,
    #     return_attention_mask=False,
    #     return_tensors='pt',
    # )
    # result = {"input_ids": tokenizer_out["input_ids"]}
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
        tokenizer: PreTrainedTokenizerBase = None,
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
