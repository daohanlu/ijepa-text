import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks

from .vision_transformer import Attention, VisionTransformer, Block, get_1d_sincos_pos_embed, \
    get_1d_sincos_pos_embed_from_grid, VisionTransformerPredictor


def get_fixed_1d_pos_embed_pytorch(max_len: int, d_model: int) -> torch.Tensor:
    """ The PyTorch implementation of 1D positional embeddings
     https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class TextTransformerPredictor(VisionTransformerPredictor):
    def __init__(self, n_positions=512, **kwargs):
        kwargs['num_patches'] = 0
        super().__init__(**kwargs)
        self.n_positions = n_positions
        del self.predictor_pos_embed
        # Changed to fixed from learnable in ViT
        self.predictor_pos_embed = get_fixed_1d_pos_embed_pytorch(self.n_positions, self.predictor_embed.out_features)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B, N, _ = x.shape
        D_predictor = self.predictor_embed.out_features

        # -- map from encoder-dim to predictor-dim and remove masked tokens
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += x_pos_embed[masks_x].reshape(B, -1, D_predictor)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = pos_embs[masks].reshape(B, -1, D)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class TextTransformer(VisionTransformer):
    """ Text Transformer using same config as Vision Transformer """

    def __init__(
            self,
            n_positions=512,
            vocab_size=32128,
            **kwargs
    ):
        kwargs['num_patches'] = 0
        super().__init__(**kwargs)
        del self.patch_embed  # this is from ViT. We don't need it.
        del self.pos_embed
        self.n_positions = n_positions
        self.token_embed = nn.Embedding(vocab_size, self.embed_dim)
        # Changed to fixed from learnable in ViT
        self.pos_embed = get_fixed_1d_pos_embed_pytorch(self.n_positions, self.embed_dim)  # (1, num_patches, embed_dim)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.token_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        assert x.shape[1] == self.pos_embed.shape[1]  # assert sequence length is equal to positional embedding size
        assert x.shape[2] == self.pos_embed.shape[2]  # assert sequence dim is equal to positional embedding dim
        x = x + self.pos_embed[:, :N, :]

        print('text_transformers: x.shape before mask', x.shape)
        # -- mask x
        if masks is not None:
            x = x[masks].reshape(B, -1, D)
        print('text_transformers: x.shape after mask', x.shape)
        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)
        else:
            raise ValueError('self.norm should not be none!')

        return x


def tet_predictor(**kwargs):
    model = TextTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def tet_tiny(patch_size=16, **kwargs):
    model = TextTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tet_small(patch_size=16, **kwargs):
    model = TextTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tet_base(patch_size=16, **kwargs):
    model = TextTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tet_large(patch_size=16, **kwargs):
    model = TextTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tet_huge(patch_size=16, **kwargs):
    model = TextTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tet_giant(patch_size=16, **kwargs):
    model = TextTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


TET_EMBED_DIMS = {
    'tet_tiny': 192,
    'tet_small': 384,
    'tet_base': 768,
    'tet_large': 1024,
    'tet_huge': 1280,
    'tet_giant': 1408,
}
