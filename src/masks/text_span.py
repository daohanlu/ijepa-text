import pdb
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding


class TextSpanCollator(object):
    pass


@dataclass
class MyDataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
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

    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    eos_token_id: int
    mask_token_id: int = 32000

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = torch.utils.data.default_collate(batch)
        input_ids = batch["input_ids"]
        # batch['unmasked_input_ids'] = input_ids.clone()
        print('batch["input_ids"] shape after collate', input_ids.shape)
        batch_size, expandend_input_length = input_ids.shape

        mask = torch.stack([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)], dim=0)
        batch['mask'] = mask  # True where tokens are masked
        batch['non_noise_mask'] = ~mask  # True where tokens are not masked
        print('mask shape', mask.shape)
        # Replace tokens with shared mask token id
        batch['input_ids'] = self.filter_input_ids(input_ids, ~mask)
        batch['labels'] = input_ids[mask].reshape(batch_size, -1)
        print('batch after extras', batch)
        print('batch["labels"] shape', batch['labels'].shape)
        print('number of masked out tokens', (batch['input_ids'] == self.mask_token_id).sum(-1))
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

    def concatenate_eos(self, input_ids):
        """ Puts an eos token at the end of every sequence """

    def filter_input_ids(self, input_ids, mask):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]
        input_ids = input_ids[mask].reshape(batch_size, -1)
        # add in eos token
        input_ids = torch.concatenate(
            [input_ids, torch.full((batch_size, 1), self.eos_token_id, dtype=torch.int32)], dim=-1
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
