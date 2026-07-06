# RandAR GPT Model
# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py

import random
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dataclasses import dataclass
from typing import Optional, List, Tuple

from .utils import DropPath, interleave_tokens, calculate_num_query_tokens_for_parallel_decoding
from .generate import sample
from .llamagen_gpt import LabelEmbedder, CaptionEmbedder, MLP, RMSNorm, \
    FeedForward, KVCache, find_multiple, apply_rotary_emb, precompute_freqs_cis_2d


def batch_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (bs, seq_len, head_dim // 2, 2)
    bs, seq_len, n_head, head_dim = x.shape
    xshaped = x.float().reshape(
        *x.shape[:-1], head_dim // 2, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        bs, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


""" Attention module modified for the parts updating KV cache
    Supporting slicing to accelerate inference
"""
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        attn_dropout_p: float,
        resid_dropout_p: float,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache_read_only: bool = False,
        prefix_len: Optional[int] = None
    ):
        """
        during inference:
        Args:
            x: [bsz, seqlen, dim], input tensor.
            freqs_cis: [bsz, seqlen, head_dim // 2, 2], used to apply rotary emb.
            input_pos: [seqlen], used to update KV cache.
            mask: [bsz, seqlen, seqlen], used to mask out attention weights.
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # this part is modified from LLaMAGen
        xq = batch_apply_rotary_emb(xq, freqs_cis)
        xk = batch_apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        is_causal = True if mask is None else False



        # this part is modified from LLaMAGen
        if self.kv_cache is not None:
            ###Do not update KV Cache for probing
            if cache_read_only:
                assert prefix_len is not None
                cached_k = self.kv_cache.k_cache[:bsz, :, :prefix_len]
                cached_v = self.kv_cache.v_cache[:bsz, :, :prefix_len]

                keys = torch.cat([cached_k, xk], dim=2)
                values = torch.cat([cached_v, xv], dim=2)
                mask = None
                is_causal = False
            else:
                # [b, n_head, max_seq_len, head_dim]
                keys, values = self.kv_cache.update(input_pos, xk, xv)

                # assuming that all the samples in a batch have the same input_pos
                max_pos = torch.max(input_pos) + 1
                keys = keys[:, :, :max_pos]
                values = values[:, :, :max_pos]
                if mask is not None:
                    mask = mask[:, :, :, :max_pos]
        else:
            keys, values = xk, xv

        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output

    def forward_with_attn(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """Like forward() but also returns averaged attention weights [bsz, n_head, q_len, kv_len]."""
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        xq = batch_apply_rotary_emb(xq, freqs_cis)
        xk = batch_apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
            max_pos = torch.max(input_pos) + 1
            keys = keys[:, :, :max_pos]
            values = values[:, :, :max_pos]
            if mask is not None:
                mask = mask[:, :, :, :max_pos]
        else:
            keys, values = xk, xv

        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # Manual attention to expose weights
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(xq, keys.transpose(-2, -1)) * scale  # [bsz, n_head, q_len, kv_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        else:
            causal_mask = torch.ones(seqlen, keys.shape[2], dtype=torch.bool, device=x.device).tril()
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [bsz, n_head, q_len, kv_len]
        output = torch.matmul(attn_weights, values)  # [bsz, n_head, q_len, head_dim]

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))
        return output, attn_weights


""" Cloned from LLaMAGen: only the attention uses our customized version
"""
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim=4096,
        n_layer=32,
        n_head=32,
        n_kv_head=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        rope_base=10000,
        norm_eps=1e-5,
        token_dropout_p=0.1,
        attn_dropout_p=0.0,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path=0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim, n_head, n_kv_head, attn_dropout_p, resid_dropout_p
        )
        self.feed_forward = FeedForward(
            dim, ffn_dim_multiplier, multiple_of, ffn_dropout_p
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
        cache_read_only: bool = False,
        prefix_len: Optional[int] = None,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, cache_read_only=cache_read_only, prefix_len=prefix_len)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class RandARTransformer(nn.Module):
    def __init__(
        self,
        dim=4096,
        n_layer=32,
        n_head=32,
        n_kv_head=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        rope_base=10000,
        norm_eps=1e-5,
        initializer_range=0.02,
        token_dropout_p=0.1,
        attn_dropout_p=0.0,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
        num_classes=1000,
        caption_dim=2048,
        class_dropout_prob=0.1,
        model_type="c2i",
        vocab_size=16384,
        cls_token_num=1,
        block_size=256,
        max_batch_size=32,
        max_seq_len=2048,
        position_order="random",
        num_inference_steps=88,
        zero_class_qk=True,
        grad_checkpointing=True,
    ):
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.token_dropout_p = token_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.drop_path_rate = drop_path_rate
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.cls_token_num = cls_token_num
        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(num_classes, dim, class_dropout_prob)
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(caption_dim, dim, class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.tok_dropout = nn.Dropout(token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(
                TransformerBlock(
                    dim=dim,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    rope_base=rope_base,
                    norm_eps=norm_eps,
                    token_dropout_p=token_dropout_p,
                    attn_dropout_p=attn_dropout_p,
                    resid_dropout_p=resid_dropout_p,
                    ffn_dropout_p=ffn_dropout_p,
                    drop_path=dpr[layer_id],
                )
            )

        # output layer
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        # initialization
        self.initializer_range = initializer_range
        self.initialize_weights()

        # RandAR related parameters
        self.pos_instruct_embeddings = nn.Parameter(torch.randn(1, self.dim) * self.initializer_range)
        self.position_order = position_order
        self.num_inference_steps = num_inference_steps
        self.zero_class_qk = zero_class_qk
        self.grad_checkpointing = grad_checkpointing

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.n_head, head_dim, dtype
            )

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

    def remove_caches(self):
        for l in self.layers:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1




    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        token_order: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None:
            return self.forward_train(idx, cond_idx, token_order, input_pos, targets, mask, valid)
        else:
            raise ValueError("idx and cond_idx cannot be both None")

    def forward_train(self,
                      idx: torch.Tensor,
                      cond_idx: torch.Tensor,
                      token_order: Optional[torch.Tensor] = None,
                      input_pos: Optional[torch.Tensor] = None,
                      targets: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None,
                      valid: Optional[torch.Tensor] = None,):
        """ Args:
            idx: [bsz, seq_len] GT image tokens for teacher forcing
            cond_idx: [bsz, cls_token_num] Cls tokens
            token_order: [bsz, seq_len] Position order for each token
            input_pos: [seq_len] Position index for each token (default None)
            targets: [bsz, seq_len] Target tokens for teacher forcing (default None)
            mask: [bsz, seq_len, seq_len] Causal mask for attention (default None)
            valid: [bsz, seq_len] Valid mask for loss calculation (default None)
        """
        # 1. Prepare orders
        bs = idx.shape[0]
        if token_order is None:
            if self.position_order == "random":
                token_order = torch.arange(self.block_size, device=self.tok_embeddings.weight.device, dtype=torch.long)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
                token_order = token_order.contiguous()
            elif self.position_order == "raster":
                token_order = torch.arange(self.block_size, device=idx.device)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)
                token_order = token_order.contiguous()
            else:
                raise ValueError(f"Invalid position order: {self.position_order}")

        # permute the image tokens according to the random order
        idx = torch.gather(idx.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous() # [bsz, seq_len]
        targets = torch.gather(targets.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous() # [bsz, seq_len]

        # 2. Prepare embeddings and freqs_cis
        self.freqs_cis = self.freqs_cis.to(cond_idx.device)
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[
            :, : self.cls_token_num
        ] # [bsz, cls_token_num, dim]

        token_embeddings = self.tok_embeddings(idx)
        token_embeddings = self.tok_dropout(token_embeddings) # [bsz, seq_len, dim]
        position_instruction_tokens = self.get_position_instruction_tokens(token_order) # [bsz, seq_len, dim]

        h = torch.cat(
            (cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)),
            dim=1
        )

        token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        freqs_cis = torch.cat(
            (self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), interleave_tokens(token_freqs_cis, token_freqs_cis)),
            dim=1
        )

        # 3. Forward
        for layer in self.layers:
            if self.grad_checkpointing:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, input_pos, mask)

        h = self.norm(h)
        logits = self.output(h).float()
        token_logits = logits[:, self.cls_token_num::2].contiguous()

        # 4. Loss computation
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), targets.view(-1))

        return token_logits, loss, token_order

    def forward_inference(self,
                          x: torch.Tensor,
                          freqs_cis: torch.Tensor,
                          input_pos: torch.Tensor):
        """ Args:
            x: [bs, query_num, dim] Input tokens
            freqs_cis: [bs, query_num, n_head, dim // n_head] Frequency embeddings
            input_pos: [query_num] Position index for each token
        """
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = self.output(h).float()
        return logits

    def forward_probe_from_cache(self, x, freqs_cis, prefix_len):
        h = x
        for layer in self.layers:
            h = layer(
                h,
                freqs_cis,
                start_pos=None,
                mask=None,
                cache_read_only=True,
                prefix_len=prefix_len,
            )
        h = self.norm(h)
        return self.output(h).float()
    def forward_inference_with_attn(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: torch.Tensor,
    ):
        """Like forward_inference() but also returns attention weights averaged over all layers and heads.

        Returns:
            logits: [bs, q_len, vocab_size]
            avg_attn: [bs, q_len, kv_len] averaged over layers and heads
        """
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        attn_sum = None
        for layer in self.layers:
            h_normed = layer.attention_norm(h)
            attn_out, attn_weights = layer.attention.forward_with_attn(h_normed, freqs_cis, input_pos, mask)
            h = h + layer.drop_path(attn_out)
            h = h + layer.drop_path(layer.feed_forward(layer.ffn_norm(h)))
            # attn_weights: [bs, n_head, q_len, kv_len] — average over heads
            layer_avg = attn_weights.mean(dim=1)  # [bs, q_len, kv_len]
            if attn_sum is None:
                attn_sum = layer_avg
            else:
                attn_sum = attn_sum + layer_avg
        avg_attn = attn_sum / len(self.layers)  # [bs, q_len, kv_len]
        h = self.norm(h)
        logits = self.output(h).float()
        return logits, avg_attn

    def get_position_instruction_tokens(self, token_order):
        position_instruct_tokens = self.pos_instruct_embeddings.view(1, 1, self.n_head, self.dim // self.n_head)
        position_instruct_tokens = position_instruct_tokens.repeat(token_order.shape[0], self.block_size, 1, 1) # [1, block_size, n_head, dim // n_head]

        # apply rotary embedding
        position_instruct_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        position_instruct_tokens = batch_apply_rotary_emb(position_instruct_tokens, position_instruct_freqs_cis)
        position_instruct_tokens = position_instruct_tokens.view(token_order.shape[0], self.block_size, self.dim).contiguous()
        return position_instruct_tokens

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def configure_optimizer(
        self, lr, weight_decay, beta1, beta2, max_grad_norm, **kwargs
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        import inspect

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            **extra_args
        )
        return optimizer

    # large changes to LLaMAGen, directly supporting parallel decoding controlled by num_inference_steps
    def generate(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """ Args:
            cond: [bsz, seq_len] Conditional tokens
            token_order: [bsz, seq_len] Position order for each token
            cfg_scales: Tuple (cfg_scale_start, cfg_scale_end) linear cfg scales, set start=end for constant cfg_scale
            num_inference_steps: int Number of inference steps, set to -1 or the number of image tokens to disable parallel decoding
            temperature: float Temperature for sampling
            top_k: int Top-k for sampling
            top_p: float Top-p for sampling
        """
        bs = cond.shape[0]

        # Step-1: Generate the token orders and result sequences
        if token_order is None:
            token_order = torch.arange(self.block_size, device=cond.device)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            token_order = token_order.contiguous()
            if self.position_order == "random":
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)

        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)

        # Step-2: Prepare the freqs_cis and position_instruction_tokens
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        img_token_freq_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]

        # Step-3: Prepare CFG
        if cfg_scales[-1] > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
            bs *= 2
        else:
            cond_combined = cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)

        # Step-4: KV Cache setup
        max_seq_len = cond_combined_tokens.shape[1] + self.block_size * 2
        with torch.device(cond.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len, dtype=self.tok_embeddings.weight.dtype)

        # Step-5: Autoregressive generation with parallel decoding
        if num_inference_steps == -1:
            # if -1, one token at a time, no parallel decoding
            num_inference_steps = self.block_size

        cur_inference_step = 0
        num_query_token_cur_step = 1 # how many tokens to decode at this step
        query_token_idx_cur_step = 0 # the index of the first token to decode at this step

        # Step 5-1: Prepare the first step
        # [cls_token, query_token_0, ..., query_token_n]
        x = torch.cat([cond_combined_tokens,
                       position_instruction_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]],
                       dim=1)
        cur_freqs_cis = torch.cat([self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
                                   img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]],
                                   dim=1)
        input_pos = torch.arange(0, x.shape[1], device=cond.device)

        # Step 5-2: Start the loop
        while query_token_idx_cur_step <= self.block_size - num_query_token_cur_step and query_token_idx_cur_step <= self.block_size - 1:
            # Step 5-3: Decode the current step tokens
            logits = self.forward_inference(x, cur_freqs_cis, input_pos)

            # apply CFG
            if cfg_scales[-1] > 1.0:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / self.block_size
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

            # query tokens' logits and indices
            logits = logits[:, -num_query_token_cur_step:] # [bs, query_num, vocab_size]
            indices = torch.zeros(result_indices.shape[0], num_query_token_cur_step, dtype=torch.long, device=cond.device)
            for i in range(num_query_token_cur_step):
                indices[:, i : i + 1] = sample(logits[:, i : i + 1], temperature=temperature, top_k=top_k, top_p=top_p)[0]

            # save the result tokens
            result_indices[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step] = indices.clone()

            img_tokens = self.tok_embeddings(indices)
            if cfg_scales[-1] > 1.0:
                img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

            # Step 5-4: Prepare for the next step
            cur_inference_step += 1
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size,
                query_token_idx_cur_step, num_query_token_cur_step)

            ########## Important: Prepare the tokens ##########
            # [cur_img_0, cur_query_1, ..., cur_query_n, cur_img_n, next_query_0, ..., next_query_m]
            x = torch.zeros(bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, self.dim, dtype=x.dtype, device=cond.device)

            # cur_img_0
            x[:, :1] = img_tokens[:, :1]

            # [cur_query_1, ..., cur_query_n]
            cur_query_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            x[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_position_instruction_tokens

            # [cur_img_1, ..., cur_img_n]
            x[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = img_tokens[:, 1 : num_query_token_cur_step]

            # [next_query_0, ..., next_query_m]
            query_token_idx_next_step = query_token_idx_cur_step + num_query_token_cur_step
            next_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            x[:, 2 * num_query_token_cur_step - 1 :] = next_position_instruction_tokens

            ########## Important: Prepare the freqs_cis ##########
            cur_freqs_cis = torch.zeros((bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, *self.freqs_cis.shape[-2:]),
                                         dtype=cur_freqs_cis.dtype, device=cond.device)

            # cur_img_0
            cur_freqs_cis[:, :1] = img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + 1]

            # [cur_query_1, ..., cur_query_n]
            cur_query_freq_cis = img_token_freq_cis[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_freq_cis

            # [cur_img_1, ..., cur_img_n]
            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = cur_query_freq_cis

            # [next_query_0, ..., next_query_m]
            next_freq_cis = img_token_freq_cis[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            cur_freqs_cis[:, 2 * num_query_token_cur_step - 1 :] = next_freq_cis

            # Step 5-5: Move the query pointer idx
            query_token_idx_cur_step = query_token_idx_next_step
            if query_token_idx_cur_step > self.block_size:
                break

            last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step] # position of cur_query_0
            input_pos = torch.arange(2 * num_query_token_cur_step - 1 + num_query_token_next_step, device=cond.device, dtype=torch.long) + last_input_pos + 1
            num_query_token_cur_step = num_query_token_next_step

        # Step 6: Return to raster order for tokenizer decoding
        reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1)
        result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)
        return result_indices

    def generate_confidence_first(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """Like generate(), but at each parallel-decoding step each image
        independently selects its num_query_token_cur_step unfilled positions
        with lowest predictive entropy (highest confidence).

        At each step a single parallel forward pass (no KV cache) is run over
        [committed_context, all_pos_instrs] to obtain per-image entropy for
        every raster position; each image then picks its own top-K lowest-
        entropy unfilled slots.  The commit forward pass mirrors generate()
        exactly.  token_order is ignored.

        Returns:
            result_indices: [bsz, block_size] sampled tokens in raster order.
        """
        bs = cond.shape[0]
        ###Creating token order just as a test:
        # Step-1: Generate the token orders and result sequences
        if token_order is None:
            token_order = torch.arange(self.block_size, device=cond.device)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            token_order = token_order.contiguous()
            if self.position_order == "random":
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)


        bs = cond.shape[0]
        original_bs = bs  # [CHANGE] keep original bs before CFG doubling

        # ------------------------------------------------------------------ #
        # [CHANGE] Step-1: Pre-compute pos_instrs and freqs for ALL raster   #
        # positions so any position can be probed or committed in any order.  #
        # generate() pre-computes these only for the fixed token_order.       #
        # ------------------------------------------------------------------ #
        raster_order   = torch.arange(self.block_size, device=cond.device).unsqueeze(0).repeat(bs, 1)
        all_pos_instrs = self.get_position_instruction_tokens(raster_order)          # [bs, block_size, dim]
        all_img_freqs  = (self.freqs_cis[self.cls_token_num:]
                          .clone().to(cond.device)
                          .unsqueeze(0).repeat(bs, 1, 1, 1))                         # [bs, block_size, h/2, 2]

        # [CHANGE] store tokens directly at raster positions; no end-reorder needed
        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)

        # [CHANGE] per-image filled mask: [original_bs, block_size]
        # BUG FIX: previously [block_size] — shared across all images, forcing
        # every image to follow the same fill order.
        filled_mask = torch.zeros(original_bs, self.block_size, dtype=torch.bool, device=cond.device)

        # Step-3: Prepare CFG  [SAME AS generate]
        use_cfg = cfg_scales[-1] > 1.0
        if use_cfg:
            cond_null     = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
            # [CHANGE] double the all-raster arrays (generate() doubled token_order-indexed arrays)
            all_img_freqs  = torch.cat([all_img_freqs,  all_img_freqs],  dim=0)
            all_pos_instrs = torch.cat([all_pos_instrs, all_pos_instrs], dim=0)
            bs *= 2
        else:
            cond_combined = cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)

        # Step-4: KV Cache setup  [SAME AS generate]
        max_seq_len = cond_combined_tokens.shape[1] + self.block_size * 2
        with torch.device(cond.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len,
                              dtype=self.tok_embeddings.weight.dtype)

        # Step-5: parallel decoding schedule  [SAME AS generate]
        if num_inference_steps == -1:
            num_inference_steps = self.block_size

        cur_inference_step       = 0
        num_query_token_cur_step = 1
        query_token_idx_cur_step = 0

        # [CHANGE] committed_ctx / committed_freqs: maintained explicitly for probe passes.
        # In generate() this state lives only implicitly inside the KV cache.
        committed_ctx   = cond_combined_tokens                                             # [bs, cls_token_num, dim]
        committed_freqs = self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1)

        ### Prime the cache with conditional tokens BEFORE the first probe pass,
        ### so probe_and_select never reads uninitialized cache slots.
        cond_freqs = self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1)
        cond_pos = torch.arange(self.cls_token_num, device=cond.device)
        _ = self.forward_inference(cond_combined_tokens, cond_freqs, cond_pos)

        # [CHANGE] cache_len tracks how many entries are actually physically
        # written into the real KV cache (as opposed to committed_ctx.shape[1],
        # which includes the current step's img-token half before it's been
        # committed via a real forward_inference call). BUG FIX: probe_and_select
        # used to read committed_ctx.shape[1] as prefix_len, which is ahead of
        # the real cache by the previous step's chunk size, so probes attended
        # to uninitialized (zero) cache slots for the tail of the context.
        cache_len = self.cls_token_num

        # ------------------------------------------------------------------ #
        # [CHANGE] Per-image gather helpers.                                  #
        # sel_bs has shape [bs, m] (per-image raster indices, CFG-doubled).  #
        # All lookups into all_pos_instrs / all_img_freqs use torch.gather   #
        # instead of plain indexing so each image selects its own positions.  #
        # BUG FIX: previously used shared 1-D cur_sel to index all images    #
        # identically, e.g. all_pos_instrs[:, cur_sel].                      #
        # ------------------------------------------------------------------ #
        freq_h, freq_w = self.freqs_cis.shape[-2], self.freqs_cis.shape[-1]

        def gather_pos(sel_bs):
            # sel_bs: [bs, m] → [bs, m, dim]
            return torch.gather(all_pos_instrs, 1,
                                sel_bs.unsqueeze(-1).expand(-1, -1, self.dim))

        def gather_freq(sel_bs):
            # sel_bs: [bs, m] → [bs, m, freq_h, freq_w]
            return torch.gather(all_img_freqs, 1,
                                sel_bs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, freq_h, freq_w))

        def to_bs(sel):
            # sel: [original_bs, m] → [bs, m] (duplicate for CFG, identity otherwise)
            # BUG FIX: CFG doubling was missing for per-image sel tensors.
            return torch.cat([sel, sel], dim=0) if use_cfg else sel

        # ------------------------------------------------------------------ #
        # [CHANGE] probe_and_select: one parallel forward pass (no KV cache)  #
        # over [committed_ctx, all_pos_instrs], then per-image entropy over   #
        # all block_size positions.  Each image independently selects its     #
        # num_to_select lowest-entropy UNFILLED positions.                    #
        # Returns sel: [original_bs, num_to_select] raster indices.          #
        #                                                                     #
        # BUG FIX: previously averaged entropy across images (mean_H) and    #
        # returned a single shared 1-D selection for all images.              #
        # ------------------------------------------------------------------ #
        def probe_and_select(num_to_select: int, step_progress: int) -> torch.Tensor:
            # unfilled positions per image: [original_bs, num_unfilled]
            unfilled = (~filled_mask).nonzero(as_tuple=False)  # [N, 2]

            # This only works cleanly if every image has same num_unfilled
            num_unfilled = (~filled_mask).sum(dim=1)[0].item()
            assert torch.all((~filled_mask).sum(dim=1) == num_unfilled)

            unfilled_sel = unfilled[:, 1].view(original_bs, num_unfilled)  # raster indices

            # duplicate for CFG if needed: [bs, num_unfilled]
            unfilled_sel_bs = to_bs(unfilled_sel)

            probe_pos   = gather_pos(unfilled_sel_bs)
            probe_freqs_only = gather_freq(unfilled_sel_bs)
            prefix_len = cache_len

            probe_logits = self.forward_probe_from_cache(probe_pos, probe_freqs_only, prefix_len=prefix_len)

            if use_cfg:
                cur_cfg = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * step_progress / self.block_size
                cond_log, uncond_log = torch.chunk(probe_logits, 2, dim=0)
                ent_logits = uncond_log + cur_cfg * (cond_log - uncond_log)
            else:
                ent_logits = probe_logits

            probs = torch.softmax(ent_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)
            # entropy: [original_bs, num_unfilled]

            _, local_topk = torch.topk(entropy, num_to_select, dim=1, largest=False)
            topk_H = torch.gather(entropy, 1, local_topk)
            local_sel = torch.gather(local_topk, 1, torch.argsort(topk_H, dim=1))

            # map local unfilled indices back to raster positions
            sel = torch.gather(unfilled_sel, 1, local_sel)  # [original_bs, num_to_select]
            return sel

        # ------------------------------------------------------------------ #
        # [CHANGE] Step 5-1: per-image probe to select the first position.   #
        # generate() uses position_instruction_tokens[:, 0] for every image. #
        # ------------------------------------------------------------------ #
        ###Switch to token order to isolate fid problem
        cur_sel = probe_and_select(num_query_token_cur_step, query_token_idx_cur_step)
        #cur_sel = token_order[:, query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step]
        # cur_sel: [original_bs, 1]
        filled_mask.scatter_(1, cur_sel, True)

        sel_bs = to_bs(cur_sel)  # [bs, 1]
        x = torch.cat([cond_combined_tokens, gather_pos(sel_bs)], dim=1)
        ###The :self.cls_token_num is the same since its irrespective of token order
        ###The gather_freq(sel_bs) gets the img_freq for the next token order selection.
        cur_freqs_cis = torch.cat([
            self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
            gather_freq(sel_bs)
        ], dim=1)
        input_pos = torch.arange(0, x.shape[1], device=cond.device)

        # Step 5-2: Start the loop  [SAME loop condition as generate]
        while (query_token_idx_cur_step <= self.block_size - num_query_token_cur_step
               and query_token_idx_cur_step <= self.block_size - 1):

            k = query_token_idx_cur_step
            m = num_query_token_cur_step

            # Step 5-3: commit forward pass  [SAME AS generate]
            logits = self.forward_inference(x, cur_freqs_cis, input_pos)
            # [CHANGE] keep cache_len in sync with what's actually been written
            # into the real KV cache by this real (non-probe) forward pass.
            cache_len = int(input_pos[-1].item()) + 1

            # apply CFG  [SAME AS generate]
            if use_cfg:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * k / self.block_size
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

            # query tokens' logits and indices  [SAME AS generate]
            logits  = logits[:, -m:]  # [original_bs, m, vocab]
            indices = torch.zeros(original_bs, m, dtype=torch.long, device=cond.device)
            for i in range(m):
                indices[:, i:i+1] = sample(logits[:, i:i+1],
                                           temperature=temperature, top_k=top_k, top_p=top_p)[0]

            # [CHANGE] scatter per-image: result_indices[b, cur_sel[b, i]] = indices[b, i]
            # BUG FIX: previously looped over a shared 1-D cur_sel and wrote the
            # same column for every image.
            result_indices.scatter_(1, cur_sel, indices)

            img_tokens = self.tok_embeddings(indices)  # [original_bs, m, dim]
            if use_cfg:
                img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

            # [CHANGE] update committed_ctx with interleaved [pos_instr_sel, img] pairs.
            # Uses per-image gather so each row reflects that image's actual selections.
            sel_bs_cur = to_bs(cur_sel)  # [bs, m]
            chunk_ctx   = torch.zeros(bs, 2 * m, self.dim,
                                      dtype=committed_ctx.dtype, device=cond.device)
            chunk_freqs = torch.zeros(bs, 2 * m, freq_h, freq_w,
                                      dtype=committed_freqs.dtype, device=cond.device)
            chunk_ctx[:, ::2]    = gather_pos(sel_bs_cur)
            chunk_ctx[:, 1::2]   = img_tokens
            chunk_freqs[:, ::2]  = gather_freq(sel_bs_cur)
            chunk_freqs[:, 1::2] = gather_freq(sel_bs_cur)
            committed_ctx   = torch.cat([committed_ctx,   chunk_ctx],   dim=1)
            committed_freqs = torch.cat([committed_freqs, chunk_freqs], dim=1)

            # Step 5-4: Prepare for the next step  [SAME AS generate]
            cur_inference_step += 1
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size, k, m)

            query_token_idx_next_step = k + m

            # [CHANGE] per-image probe to select the next batch of positions
            if query_token_idx_next_step < self.block_size and num_query_token_next_step > 0:
                #To isolate the issue CHANGE BACK LATER
                next_sel = probe_and_select(num_query_token_next_step, query_token_idx_next_step)
                #next_sel = token_order[:, query_token_idx_next_step: query_token_idx_next_step + num_query_token_next_step]
                # next_sel: [original_bs, m_next]
                filled_mask.scatter_(1, next_sel, True)
            else:
                next_sel = torch.empty(original_bs, 0, dtype=torch.long, device=cond.device)

            ########## Important: Prepare the tokens ##########
            # [SAME structure as generate; gathering per-image via gather_pos
            #  instead of token_order slices into position_instruction_tokens]
            x = torch.zeros(bs, 2 * m - 1 + num_query_token_next_step, self.dim,
                            dtype=x.dtype, device=cond.device)

            # cur_img_0  [SAME]
            x[:, :1] = img_tokens[:, :1]

            # [cur_query_1, ..., cur_query_m-1]  [CHANGE]
            if m > 1:
                x[:, 1:2*m-1][:, ::2]  = gather_pos(to_bs(cur_sel[:, 1:]))
                x[:, 1:2*m-1][:, 1::2] = img_tokens[:, 1:m]

            # [next_query_0, ..., next_query_m_next-1]  [CHANGE]
            if num_query_token_next_step > 0:
                x[:, 2*m-1:] = gather_pos(to_bs(next_sel))

            ########## Important: Prepare the freqs_cis ##########
            # [SAME structure as generate; same gather-based index change]
            cur_freqs_cis = torch.zeros((bs, 2 * m - 1 + num_query_token_next_step, freq_h, freq_w),
                                        dtype=cur_freqs_cis.dtype, device=cond.device)

            # cur_img_0  [CHANGE]
            cur_freqs_cis[:, :1] = gather_freq(to_bs(cur_sel[:, :1]))

            # [cur_query_1, ..., cur_query_m-1]  [CHANGE]
            if m > 1:
                cur_freqs_cis[:, 1:2*m-1][:, ::2]  = gather_freq(to_bs(cur_sel[:, 1:]))
                cur_freqs_cis[:, 1:2*m-1][:, 1::2] = gather_freq(to_bs(cur_sel[:, 1:]))

            # [next_query_0, ..., next_query_m_next-1]  [CHANGE]
            if num_query_token_next_step > 0:
                cur_freqs_cis[:, 2*m-1:] = gather_freq(to_bs(next_sel))

            # Step 5-5: Move the query pointer idx  [SAME AS generate]
            query_token_idx_cur_step = query_token_idx_next_step
            if query_token_idx_cur_step > self.block_size:
                break

            last_input_pos = input_pos[input_pos.shape[0] - m]
            input_pos = (torch.arange(2 * m - 1 + num_query_token_next_step,
                                      device=cond.device, dtype=torch.long)
                         + last_input_pos + 1)
            num_query_token_cur_step = num_query_token_next_step
            cur_sel = next_sel

        # Step 6: result_indices already in raster order  [CHANGE: no argsort/gather needed]
        return result_indices

    def generate_beam(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor,
        beam_size: int = 4,
        num_branches: Optional[int] = None,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """Beam search over which positions to decode next, layered on top of the
        same confidence-driven position selection used by generate_confidence_first.

        At each parallel-decoding step, probe_and_select_chunks() ranks all unfilled
        raster positions by entropy (one cache-free forward pass per beam over
        [committed_ctx, all_pos_instrs], exactly like generate_confidence_first's
        probe_and_select) and proposes num_branches chunk options per beam: option 0
        is the lowest-entropy group of m positions (the same choice
        generate_confidence_first would make), and option i>0 swaps exactly slot
        i-1 of that group for the next-best unfilled candidate.

        Unlike the probe pass, each (beam, option) candidate is then scored with its
        own real commit forward pass -- batched but still one full forward per
        candidate, no KV cache -- over just [committed_ctx, chosen_chunk_query_
        tokens_in_rank_order]. This mirrors the causal structure that
        generate_confidence_first's KV-cached commit step produces (a chunk's query
        tokens only ever attend to already-committed content and to each other, in
        rank order -- never to other unfilled raster positions), so branches are
        compared on genuinely-computed logits rather than the probe's approximate
        ones. Tokens are sampled per candidate and their log-probs accumulate into
        that beam's running score. After scoring, only the top beam_size (beam,
        option) pairs per image survive to the next step.

        token_order is ignored (positions are chosen by confidence, like
        generate_confidence_first).

        Returns:
            result_indices: [bsz, block_size] tokens of the best-scoring beam per
                            image, in raster order.
        """
        O = cond.shape[0]  # original batch size
        B = beam_size
        num_branches = num_branches if num_branches is not None else beam_size
        use_cfg = cfg_scales[-1] > 1.0
        D = 2 if use_cfg else 1

        # generate_beam never calls setup_caches (branching/pruning would require
        # re-indexing a KV cache, which we're deliberately avoiding) -- so any cache
        # left behind by a previous generate()/generate_confidence_first() call on
        # this model must be disabled for the duration, then restored.
        saved_caches = [l.attention.kv_cache for l in self.layers]
        for l in self.layers:
            l.attention.kv_cache = None

        # ------------------------------------------------------------------ #
        # Step-1: all-raster pos_instrs / freqs, expanded across the beam dim. #
        # Content is identical across beams (positions don't depend on beam),  #
        # only which subset each beam later selects differs.                  #
        # ------------------------------------------------------------------ #
        raster_order   = torch.arange(self.block_size, device=cond.device).unsqueeze(0).repeat(O * B, 1)
        all_pos_instrs = self.get_position_instruction_tokens(raster_order)              # [O*B, block_size, dim]
        all_img_freqs  = (self.freqs_cis[self.cls_token_num:]
                          .clone().to(cond.device)
                          .unsqueeze(0).repeat(O * B, 1, 1, 1))                          # [O*B, block_size, h/2, 2]

        result_indices = torch.zeros((O, B, self.block_size), dtype=torch.long, device=cond.device)
        filled_mask    = torch.zeros((O, B, self.block_size), dtype=torch.bool, device=cond.device)
        beam_logprob   = torch.zeros((O, B), dtype=torch.float32, device=cond.device)

        # Step-2: Prepare CFG  [SAME idea AS generate_confidence_first, applied after beam expansion]
        cond_beam = cond.repeat_interleave(B, dim=0)                                     # [O*B, ...] image-major, beam-minor
        if use_cfg:
            cond_null      = torch.ones_like(cond_beam) * self.num_classes
            cond_combined  = torch.cat([cond_beam, cond_null])
            all_img_freqs  = torch.cat([all_img_freqs,  all_img_freqs],  dim=0)
            all_pos_instrs = torch.cat([all_pos_instrs, all_pos_instrs], dim=0)
        else:
            cond_combined = cond_beam
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)            # [O*B*D, cls_token_num, dim]

        freq_h, freq_w = self.freqs_cis.shape[-2], self.freqs_cis.shape[-1]

        # committed_ctx / committed_freqs: full sequence built so far for every
        # (image, beam) pair, CFG-doubled. No KV cache -- every step replays this
        # whole sequence from scratch, which is what keeps branching/pruning simple.
        committed_ctx   = cond_combined_tokens
        committed_freqs = self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(O * B * D, 1, 1, 1)

        def gather_pos(flat_sel, pos_bank):
            # flat_sel: [n, m] raster indices, pos_bank: [n, block_size, dim] -> [n, m, dim]
            return torch.gather(pos_bank, 1, flat_sel.unsqueeze(-1).expand(-1, -1, self.dim))

        def gather_freq(flat_sel, freq_bank):
            return torch.gather(freq_bank, 1,
                                flat_sel.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, freq_h, freq_w))

        def expand_for_options(bank, n_opt):
            # bank: [O*B*D, ...] -> [O*B*n_opt*D, ...], preserving the (cond-block,
            # uncond-block) split and (image, beam)-major / option-minor ordering.
            if use_cfg:
                cond_bank, uncond_bank = bank.chunk(2, dim=0)
                return torch.cat([cond_bank.repeat_interleave(n_opt, dim=0),
                                  uncond_bank.repeat_interleave(n_opt, dim=0)], dim=0)
            return bank.repeat_interleave(n_opt, dim=0)

        def regather_beams(bank, parent_flat):
            # bank: [O*B*D, ...] -> reordered/duplicated to the O*B surviving rows
            if use_cfg:
                cond_bank, uncond_bank = bank.chunk(2, dim=0)
                return torch.cat([cond_bank[parent_flat], uncond_bank[parent_flat]], dim=0)
            return bank[parent_flat]

        # ------------------------------------------------------------------ #
        # probe_and_select_chunks: one cache-free forward pass per beam over    #
        # [committed_ctx, all_pos_instrs] to rank unfilled positions by entropy #
        # (same heuristic generate_confidence_first's probe uses for ranking   #
        # only, never for final logits). Builds num_branches candidate chunks: #
        # option 0 is the base lowest-entropy group of m positions; option i>0 #
        # swaps slot i-1 of that group for the next-best unused candidate.     #
        #                                                                      #
        # Returns sel: [O, B, n_opt, m] raster indices.                        #
        # ------------------------------------------------------------------ #
        def probe_and_select_chunks(m: int, step_progress: int, n_opt: int) -> torch.Tensor:
            probe_x     = torch.cat([committed_ctx,   all_pos_instrs], dim=1)
            probe_freqs = torch.cat([committed_freqs, all_img_freqs],  dim=1)

            h = probe_x
            for layer in self.layers:
                h = layer(h, probe_freqs, start_pos=None, mask=None)
            h = self.norm(h)
            probe_logits = self.output(h).float()[:, -self.block_size:]  # [O*B*D, block_size, vocab]

            if use_cfg:
                cur_cfg = (cfg_scales[0]
                           + (cfg_scales[-1] - cfg_scales[0]) * step_progress / self.block_size)
                cond_log, uncond_log = torch.chunk(probe_logits, 2, dim=0)
                ent_logits = uncond_log + cur_cfg * (cond_log - uncond_log)  # [O*B, block_size, vocab]
            else:
                ent_logits = probe_logits

            probs   = torch.softmax(ent_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)  # [O*B, block_size]
            entropy = entropy.view(O, B, self.block_size).masked_fill(filled_mask, float('inf'))

            n_total = m + n_opt - 1
            _, idx_topk = torch.topk(entropy, n_total, dim=-1, largest=False)           # [O, B, n_total]
            vals_topk   = torch.gather(entropy, -1, idx_topk)
            ranked      = torch.gather(idx_topk, -1, torch.argsort(vals_topk, dim=-1))  # ascending entropy

            base  = ranked[..., :m]  # [O, B, m]
            extra = ranked[..., m:]  # [O, B, n_opt - 1]

            sel = base.unsqueeze(2).repeat(1, 1, n_opt, 1).clone()  # [O, B, n_opt, m]
            for i in range(1, n_opt):
                sel[:, :, i, i - 1] = extra[:, :, i - 1]
            return sel

        # Step-5: parallel decoding schedule  [SAME AS generate_confidence_first]
        if num_inference_steps == -1:
            num_inference_steps = self.block_size

        cur_inference_step       = 0
        num_query_token_cur_step = 1
        query_token_idx_cur_step = 0

        while (query_token_idx_cur_step <= self.block_size - num_query_token_cur_step
               and query_token_idx_cur_step <= self.block_size - 1):

            k = query_token_idx_cur_step
            m = num_query_token_cur_step

            # clamp branching factor if too few unfilled positions remain for extra candidates
            n_opt = max(1, min(num_branches, self.block_size - k - m + 1))

            sel = probe_and_select_chunks(m, k, n_opt)  # [O, B, n_opt, m]
            sel_flat = sel.view(O * B * n_opt, m)       # (image, beam, option)-major

            # ------------------------------------------------------------------ #
            # Real commit forward per (beam, option): batch = O*B*n_opt*D, over   #
            # [committed_ctx repeated per option, chosen chunk's query tokens in   #
            # rank order]. This mirrors the causal structure of                   #
            # generate_confidence_first's KV-cached commit step, just recomputed   #
            # from scratch each step instead of incrementally cached.             #
            # ------------------------------------------------------------------ #
            sel_flat_d  = torch.cat([sel_flat, sel_flat], dim=0) if use_cfg else sel_flat  # [O*B*n_opt*D, m]
            chunk_pos   = gather_pos(sel_flat_d, expand_for_options(all_pos_instrs, n_opt))
            chunk_freqs = gather_freq(sel_flat_d, expand_for_options(all_img_freqs, n_opt))

            commit_x     = torch.cat([expand_for_options(committed_ctx, n_opt),   chunk_pos],   dim=1)
            commit_freqs = torch.cat([expand_for_options(committed_freqs, n_opt), chunk_freqs], dim=1)

            h = commit_x
            for layer in self.layers:
                h = layer(h, commit_freqs, start_pos=None, mask=None)
            h = self.norm(h)
            commit_logits = self.output(h).float()[:, -m:]  # [O*B*n_opt*D, m, vocab]

            if use_cfg:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * k / self.block_size
                cond_logits, uncond_logits = torch.chunk(commit_logits, 2, dim=0)
                commit_logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)  # [O*B*n_opt, m, vocab]

            # sample each of the m chunk positions per candidate and accumulate log-probs
            n_cand        = O * B * n_opt
            indices       = torch.zeros(n_cand, m, dtype=torch.long, device=cond.device)
            chunk_logprob = torch.zeros(n_cand, device=cond.device)
            for i in range(m):
                idx_i, probs_i = sample(commit_logits[:, i:i+1], temperature=temperature, top_k=top_k, top_p=top_p)
                indices[:, i:i+1] = idx_i
                chunk_logprob += torch.log(torch.gather(probs_i, 1, idx_i).clamp(min=1e-10)).squeeze(-1)

            # ------------------------------------------------------------------ #
            # Prune: per image, keep the top beam_size (beam, option) candidates   #
            # by cumulative log-prob.                                             #
            # ------------------------------------------------------------------ #
            cand_logprob = beam_logprob.unsqueeze(-1).repeat(1, 1, n_opt).view(O, B * n_opt) \
                           + chunk_logprob.view(O, B * n_opt)

            keep_vals, keep_idx = torch.topk(cand_logprob, B, dim=1, largest=True)  # [O, B]
            parent_beam   = keep_idx // n_opt  # [O, B] which OLD beam (0..B-1) each survivor came from
            # (chosen_option = keep_idx % n_opt is implicit in the gathers below)

            sel_o   = sel.view(O, B * n_opt, m)      # [O, B*n_opt, m]
            idx_o   = indices.view(O, B * n_opt, m)  # [O, B*n_opt, m]
            new_sel = torch.gather(sel_o, 1, keep_idx.unsqueeze(-1).expand(-1, -1, m))  # [O, B, m]
            new_tok = torch.gather(idx_o, 1, keep_idx.unsqueeze(-1).expand(-1, -1, m))  # [O, B, m]

            # rebuild per-beam state from the surviving parents, then record this step's chunk
            result_indices = torch.gather(result_indices, 1,
                                          parent_beam.unsqueeze(-1).expand(-1, -1, self.block_size)).clone()
            filled_mask    = torch.gather(filled_mask, 1,
                                          parent_beam.unsqueeze(-1).expand(-1, -1, self.block_size)).clone()
            result_indices.scatter_(2, new_sel, new_tok)
            filled_mask.scatter_(2, new_sel, True)
            beam_logprob = keep_vals

            # rebuild committed_ctx / committed_freqs for surviving beams, then append this chunk
            parent_flat = (torch.arange(O, device=cond.device).unsqueeze(1) * B + parent_beam).view(-1)  # [O*B]
            committed_ctx   = regather_beams(committed_ctx,   parent_flat)
            committed_freqs = regather_beams(committed_freqs, parent_flat)

            new_sel_flat   = new_sel.view(O * B, m)
            new_tok_flat   = new_tok.view(O * B, m)
            new_sel_flat_d = torch.cat([new_sel_flat, new_sel_flat], dim=0) if use_cfg else new_sel_flat

            new_pos  = gather_pos(new_sel_flat_d, all_pos_instrs)  # [O*B*D, m, dim]
            new_freq = gather_freq(new_sel_flat_d, all_img_freqs)  # [O*B*D, m, h, w]
            new_img  = self.tok_embeddings(new_tok_flat)           # [O*B, m, dim]
            if use_cfg:
                new_img = torch.cat([new_img, new_img], dim=0)     # [O*B*D, m, dim]

            chunk_ctx   = torch.zeros(O * B * D, 2 * m, self.dim, dtype=committed_ctx.dtype, device=cond.device)
            chunk_freqs = torch.zeros(O * B * D, 2 * m, freq_h, freq_w, dtype=committed_freqs.dtype, device=cond.device)
            chunk_ctx[:, ::2]    = new_pos
            chunk_ctx[:, 1::2]   = new_img
            chunk_freqs[:, ::2]  = new_freq
            chunk_freqs[:, 1::2] = new_freq
            committed_ctx   = torch.cat([committed_ctx,   chunk_ctx],   dim=1)
            committed_freqs = torch.cat([committed_freqs, chunk_freqs], dim=1)

            # Step 5-4: advance the shared schedule  [SAME AS generate_confidence_first]
            cur_inference_step += 1
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size, k, m)
            query_token_idx_cur_step = k + m
            num_query_token_cur_step = num_query_token_next_step

        for l, c in zip(self.layers, saved_caches):
            l.attention.kv_cache = c

        # Step-6: pick the best-scoring beam per image; result_indices is already in raster order
        best_beam = torch.argmax(beam_logprob, dim=1)  # [O]
        result_indices = torch.gather(result_indices, 1,
                                      best_beam.view(O, 1, 1).expand(-1, 1, self.block_size)).squeeze(1)
        return result_indices

    def generate_with_entropy(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """ Same as generate, but also returns the per-token entropy of the post-CFG logits.

        Returns:
            result_indices: [bsz, block_size] sampled token indices in raster order
            result_entropy: [bsz, block_size] entropy (in nats) of the logit distribution, in raster order
        """
        bs = cond.shape[0]

        # Step-1: Generate the token orders and result sequences
        if token_order is None:
            token_order = torch.arange(self.block_size, device=cond.device)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            token_order = token_order.contiguous()
            if self.position_order == "random":
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)

        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)
        result_entropy = torch.zeros((bs, self.block_size), dtype=torch.float, device=cond.device)

        # Step-2: Prepare the freqs_cis and position_instruction_tokens
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        img_token_freq_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]

        # Step-3: Prepare CFG
        if cfg_scales[-1] > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
            bs *= 2
        else:
            cond_combined = cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)

        # Step-4: KV Cache setup
        max_seq_len = cond_combined_tokens.shape[1] + self.block_size * 2
        with torch.device(cond.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len, dtype=self.tok_embeddings.weight.dtype)

        # Step-5: Autoregressive generation with parallel decoding
        if num_inference_steps == -1:
            num_inference_steps = self.block_size

        cur_inference_step = 0
        num_query_token_cur_step = 1
        query_token_idx_cur_step = 0

        # Step 5-1: Prepare the first step
        x = torch.cat([cond_combined_tokens,
                       position_instruction_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]],
                       dim=1)
        cur_freqs_cis = torch.cat([self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
                                   img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]],
                                   dim=1)
        input_pos = torch.arange(0, x.shape[1], device=cond.device)

        # Step 5-2: Start the loop
        while query_token_idx_cur_step <= self.block_size - num_query_token_cur_step and query_token_idx_cur_step <= self.block_size - 1:
            # Step 5-3: Decode the current step tokens
            logits = self.forward_inference(x, cur_freqs_cis, input_pos)

            # apply CFG
            if cfg_scales[-1] > 1.0:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / self.block_size
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

            # query tokens' logits and indices
            logits = logits[:, -num_query_token_cur_step:]  # [bs, query_num, vocab_size]

            # store per-token entropy for this batch of query positions
            probs = torch.softmax(logits, dim=-1)
            token_entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)
            result_entropy[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step] = token_entropy

            indices = torch.zeros(result_indices.shape[0], num_query_token_cur_step, dtype=torch.long, device=cond.device)
            for i in range(num_query_token_cur_step):
                indices[:, i : i + 1] = sample(logits[:, i : i + 1], temperature=temperature, top_k=top_k, top_p=top_p)[0]

            # save the result tokens
            result_indices[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step] = indices.clone()

            img_tokens = self.tok_embeddings(indices)
            if cfg_scales[-1] > 1.0:
                img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

            # Step 5-4: Prepare for the next step
            cur_inference_step += 1
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size,
                query_token_idx_cur_step, num_query_token_cur_step)

            ########## Important: Prepare the tokens ##########
            x = torch.zeros(bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, self.dim, dtype=x.dtype, device=cond.device)

            x[:, :1] = img_tokens[:, :1]

            cur_query_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            x[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_position_instruction_tokens

            x[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = img_tokens[:, 1 : num_query_token_cur_step]

            query_token_idx_next_step = query_token_idx_cur_step + num_query_token_cur_step
            next_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            x[:, 2 * num_query_token_cur_step - 1 :] = next_position_instruction_tokens

            ########## Important: Prepare the freqs_cis ##########
            cur_freqs_cis = torch.zeros((bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, *self.freqs_cis.shape[-2:]),
                                         dtype=cur_freqs_cis.dtype, device=cond.device)

            cur_freqs_cis[:, :1] = img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + 1]

            cur_query_freq_cis = img_token_freq_cis[:, query_token_idx_cur_step + 1 : query_token_idx_cur_step + num_query_token_cur_step]
            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_freq_cis

            cur_freqs_cis[:, 1 : 2 * num_query_token_cur_step - 1][:, 1::2] = cur_query_freq_cis

            next_freq_cis = img_token_freq_cis[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            cur_freqs_cis[:, 2 * num_query_token_cur_step - 1 :] = next_freq_cis

            # Step 5-5: Move the query pointer idx
            query_token_idx_cur_step = query_token_idx_next_step
            if query_token_idx_cur_step > self.block_size:
                break

            last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step]
            input_pos = torch.arange(2 * num_query_token_cur_step - 1 + num_query_token_next_step, device=cond.device, dtype=torch.long) + last_input_pos + 1
            num_query_token_cur_step = num_query_token_next_step

        # Step 6: Return to raster order for tokenizer decoding
        reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1)
        result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)
        result_entropy = torch.gather(result_entropy, 1, reverse_permutation.squeeze(-1))
        return result_indices, result_entropy

    def generate_with_attention(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """Like generate() but also returns per-token attention over previously revealed tokens.

        Attention is averaged over all layers and all heads. For each generated token at
        generation step i, we record its attention (summed over the [pos_instr, img] KV pair)
        to each of the i preceding context image tokens.

        Returns:
            result_indices:    [original_bs, block_size] sampled tokens in raster order
            result_attentions: list[original_bs] of list[block_size] of 1-D float tensors.
                               result_attentions[b][i] has length i: the attention that
                               generation-step i's query paid to generation steps 0..i-1.
                               Stored in generation order; use token_order to map to raster.
        """
        original_bs = cond.shape[0]
        bs = original_bs
        use_cfg = cfg_scales[-1] > 1.0

        # Step-1: token order
        if token_order is None:
            token_order = torch.arange(self.block_size, device=cond.device)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            if self.position_order == "random":
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)

        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)

        # result_attentions[b][gen_step] = 1-D tensor of length gen_step
        result_attentions = [[torch.zeros(0, dtype=torch.float32) for _ in range(self.block_size)]
                             for _ in range(bs)]

        # Step-2: embeddings
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        img_token_freq_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]

        # Step-3: CFG
        if use_cfg:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
            bs *= 2
        else:
            cond_combined = cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)

        # Step-4: KV cache
        max_seq_len = cond_combined_tokens.shape[1] + self.block_size * 2
        with torch.device(cond.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len, dtype=self.tok_embeddings.weight.dtype)

        if num_inference_steps == -1:
            num_inference_steps = self.block_size

        C = self.cls_token_num  # KV offset: pos_q[i] lives at C+2*i, img[i] at C+2*i+1

        cur_inference_step = 0
        num_query_token_cur_step = 1
        query_token_idx_cur_step = 0

        x = torch.cat([cond_combined_tokens,
                       position_instruction_tokens[:, 0:1]], dim=1)
        cur_freqs_cis = torch.cat([self.freqs_cis[:C].unsqueeze(0).repeat(bs, 1, 1, 1),
                                   img_token_freq_cis[:, 0:1]], dim=1)
        input_pos = torch.arange(0, x.shape[1], device=cond.device)

        while query_token_idx_cur_step <= self.block_size - num_query_token_cur_step and query_token_idx_cur_step <= self.block_size - 1:
            k = query_token_idx_cur_step
            m = num_query_token_cur_step

            logits, avg_attn = self.forward_inference_with_attn(x, cur_freqs_cis, input_pos)
            # avg_attn: [bs, q_len, kv_len]

            # CFG
            if use_cfg:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * k / self.block_size
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)
                # Use conditional-batch attention for visualization
                avg_attn = avg_attn[:original_bs]

            logits = logits[:, -m:]
            indices = torch.zeros(original_bs, m, dtype=torch.long, device=cond.device)
            for i in range(m):
                indices[:, i:i+1] = sample(logits[:, i:i+1], temperature=temperature, top_k=top_k, top_p=top_p)[0]

            result_indices[:, k:k+m] = indices.clone()

            # Extract attention for each query token in this step
            # Query for gen step k+j is at x position q_len-m+j
            if k > 0:
                ctx_kv_pos_q = torch.arange(k, device=cond.device) * 2 + C    # [C, C+2, ..., C+2*(k-1)]
                ctx_kv_pos_img = ctx_kv_pos_q + 1                              # [C+1, C+3, ..., C+2*(k-1)+1]
                for j in range(m):
                    q_idx = avg_attn.shape[1] - m + j  # position of this query in avg_attn
                    attn_to_ctx = avg_attn[:, q_idx, ctx_kv_pos_q] + avg_attn[:, q_idx, ctx_kv_pos_img]  # [bs_orig, k]
                    for b in range(original_bs):
                        result_attentions[b][k + j] = attn_to_ctx[b].cpu()

            img_tokens = self.tok_embeddings(indices)
            if use_cfg:
                img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

            cur_inference_step += 1
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size,
                k, m)

            x = torch.zeros(bs, 2 * m - 1 + num_query_token_next_step, self.dim, dtype=x.dtype, device=cond.device)
            x[:, :1] = img_tokens[:, :1]
            cur_query_position_instruction_tokens = position_instruction_tokens[:, k+1:k+m]
            x[:, 1:2*m-1][:, ::2] = cur_query_position_instruction_tokens
            x[:, 1:2*m-1][:, 1::2] = img_tokens[:, 1:m]
            query_token_idx_next_step = k + m
            next_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_next_step:query_token_idx_next_step+num_query_token_next_step]
            x[:, 2*m-1:] = next_position_instruction_tokens

            cur_freqs_cis = torch.zeros((bs, 2*m-1+num_query_token_next_step, *self.freqs_cis.shape[-2:]),
                                        dtype=cur_freqs_cis.dtype, device=cond.device)
            cur_freqs_cis[:, :1] = img_token_freq_cis[:, k:k+1]
            cur_query_freq_cis = img_token_freq_cis[:, k+1:k+m]
            cur_freqs_cis[:, 1:2*m-1][:, ::2] = cur_query_freq_cis
            cur_freqs_cis[:, 1:2*m-1][:, 1::2] = cur_query_freq_cis
            next_freq_cis = img_token_freq_cis[:, query_token_idx_next_step:query_token_idx_next_step+num_query_token_next_step]
            cur_freqs_cis[:, 2*m-1:] = next_freq_cis

            query_token_idx_cur_step = query_token_idx_next_step
            if query_token_idx_cur_step > self.block_size:
                break

            last_input_pos = input_pos[input_pos.shape[0] - m]
            input_pos = torch.arange(2*m-1+num_query_token_next_step, device=cond.device, dtype=torch.long) + last_input_pos + 1
            num_query_token_cur_step = num_query_token_next_step

        # Reorder indices to raster order
        reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1)
        result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)

        return result_indices, result_attentions

    def generate_parallel(
        self,
        cond: torch.Tensor,
        token_order: torch.Tensor,
        cfg_scales: Tuple[float, float] = (1.0, 1.0),
        num_inference_steps: int = 88,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """ Same signature as generate, but decodes all tokens in a single parallel
            forward pass (no KV cache, no iterative loop).

            Returns:
                result_indices: [bsz, block_size] sampled token indices in raster order
                result_logits:  [bsz, block_size, vocab_size] logits in raster order
        """
        bs = cond.shape[0]

        # Step-1: Generate token order
        if token_order is None:
            token_order = torch.arange(self.block_size, device=cond.device)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            token_order = token_order.contiguous()
            if self.position_order == "random":
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)

        # Step-2: Prepare embeddings and freqs_cis
        self.freqs_cis = self.freqs_cis.to(cond.device)
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        img_token_freq_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]

        # Step-3: Prepare CFG
        use_cfg = cfg_scales[-1] > 1.0
        if use_cfg:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
        else:
            cond_combined = cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)
        effective_bs = cond_combined_tokens.shape[0]

        # Step-4: Single parallel forward pass — no KV cache
        x = torch.cat([cond_combined_tokens, position_instruction_tokens], dim=1)
        cur_freqs_cis = torch.cat([
            self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(effective_bs, 1, 1, 1),
            img_token_freq_cis,
        ], dim=1)

        # Temporarily disable KV caches so attention runs as plain causal (is_causal=True)
        saved_caches = [layer.attention.kv_cache for layer in self.layers]
        for layer in self.layers:
            layer.attention.kv_cache = None

        h = x
        for layer in self.layers:
            h = layer(h, cur_freqs_cis, start_pos=None, mask=None)
        h = self.norm(h)
        logits = self.output(h).float()

        for layer, cache in zip(self.layers, saved_caches):
            layer.attention.kv_cache = cache

        # Step-5: Apply CFG with per-token linearly varying scale
        token_logits = logits[:, self.cls_token_num:]  # [effective_bs, block_size, vocab_size]
        if use_cfg:
            cfg_scale_per_token = torch.linspace(cfg_scales[0], cfg_scales[-1], self.block_size, device=cond.device)
            cfg_scale_per_token = cfg_scale_per_token.view(1, self.block_size, 1)
            cond_logits, uncond_logits = torch.chunk(token_logits, 2, dim=0)
            token_logits = uncond_logits + cfg_scale_per_token * (cond_logits - uncond_logits)

        # Step-6: Sample tokens
        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)
        for i in range(self.block_size):
            result_indices[:, i:i+1] = sample(token_logits[:, i:i+1], temperature=temperature, top_k=top_k, top_p=top_p)[0]

        # Step-7: Return to raster order
        reverse_permutation = torch.argsort(token_order, dim=-1).long()
        result_indices = torch.gather(result_indices, 1, reverse_permutation)
        result_logits = torch.gather(
            token_logits, 1,
            reverse_permutation.unsqueeze(-1).expand(-1, -1, self.vocab_size),
        )

        return result_indices, result_logits
