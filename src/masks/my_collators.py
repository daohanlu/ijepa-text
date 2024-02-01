#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
import pdb
from dataclasses import asdict, dataclass, field

from typing import Dict, List, Optional

import datasets
import numpy as np
import torch

from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

from src.datasets.c4 import get_tokenizer, tokenize_function


def append_eos(input_ids: torch.Tensor, eos_token_id) -> torch.Tensor:
    batch_size = input_ids.shape[0]
    return torch.concatenate(
        [input_ids, torch.full((batch_size, 1), eos_token_id, dtype=torch.int32)], dim=-1
    )


def append_false(mask_indices: torch.Tensor) -> torch.Tensor:
    batch_size = mask_indices.shape[0]
    return torch.concatenate(
        [mask_indices, torch.full((batch_size, 1), False, dtype=torch.bool)], dim=-1
    )


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

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

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        # Modified for the case where we replace each masked token with a shared mask token in the input
        _input_length = tokens_length + 1
        _output_length = num_noise_tokens + 1
        return _input_length, _output_length

    tokens_length = inputs_length - 10

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@dataclass
class MyDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        batch = torch.utils.data.default_collate(examples)
        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = torch.stack([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        # We append False to the noise mask to indicate we never mask the ending EOS token
        batch['noise_mask'] = append_false(mask_indices)  # True where tokens are masked
        batch['non_noise_mask'] = append_false(~mask_indices)  # True where tokens are not masked
        batch['input_ids'] = append_eos(input_ids, self.tokenizer.eos_token_id)
        labels = input_ids[mask_indices].reshape(batch_size, -1)
        batch['labels'] = append_eos(labels, self.tokenizer.eos_token_id)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return torch.from_numpy(is_noise[:orig_length])


def main():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = get_tokenizer()
    num_workers = 8
    # tokenizer.mask_token = 32000
    input_length = 512
    mlm_probability = 0.15
    mean_noise_span_length = 3.0
    before_mask_input_length, target_length = compute_input_and_target_lengths(
        inputs_length=input_length,
        noise_density=mlm_probability,
        mean_noise_span_length=mean_noise_span_length,
    )
    print(f'input_length: {input_length}, mlm_probability: {mlm_probability}, '
          f'mean_noise_span_length {mean_noise_span_length}.')
    print(f'Computed before_mask_input_length: {before_mask_input_length}, target_length: {target_length}')
    # --- Load C4 dataset ---
    print('Loading C4 dataset (train split, en) with streaming=True')
    dataset = datasets.load_dataset('allenai/c4', 'en', split='train', streaming=True)
    dataset = dataset.remove_columns(
        ['timestamp', 'url']
    )
    assert dataset.n_shards >= 1024, \
        ("We want to have >=1024 shards for efficient processing with num_workers in PyTorch dataloader. "
         "The number of shards is: " + str(dataset.n_shards))
    # ---Tokenize and shuffle---
    print(f"tokenizer is {tokenizer.__class__} and tokenizer.pad_token_id is {tokenizer.pad_token_id}."
          f"tokenizer.eos_token_id is {tokenizer.eos_token_id}")

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
    collator = MyDataCollatorForT5MLM(tokenizer=tokenizer,
                                      noise_density=mlm_probability,
                                      mean_noise_span_length=mean_noise_span_length,
                                      input_length=input_length,
                                      target_length=target_length,
                                      pad_token_id=tokenizer.pad_token_id,
                                      decoder_start_token_id=tokenizer.pad_token_id, )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=8,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=False)
    for itr, batch in enumerate(data_loader):
        print('batch keys' + str(list(batch.keys())))
        pdb.set_trace()


if __name__ == '__main__':
    main()
