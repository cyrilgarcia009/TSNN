import copy
from typing import Optional
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def keep_topk_per_row(x: torch.Tensor, k: int = 2) -> torch.Tensor:
    vals, idx = torch.topk(x, k=k, dim=-1, largest=True)
    out = torch.zeros_like(x)
    out.scatter_(-1, idx, 1)
    return out


_attn_weights = []
_attn_softmax = []


def _stabilize_attention_logits(attn_weight: torch.Tensor) -> torch.Tensor:
    """
    Ensure every attention row has at least one finite logit before softmax.
    Fully masked rows otherwise become all -inf and produce NaNs.
    """
    finite_mask = torch.isfinite(attn_weight)
    valid_row = finite_mask.any(dim=-1, keepdim=True)
    if valid_row.all():
        return attn_weight

    fallback = torch.zeros_like(attn_weight)
    if attn_weight.size(-1) == attn_weight.size(-2):
        eye = torch.eye(attn_weight.size(-1), dtype=torch.bool, device=attn_weight.device)
        while eye.dim() < attn_weight.dim():
            eye = eye.unsqueeze(0)
        fallback = fallback.masked_fill(~eye, float("-inf"))

    return torch.where(valid_row, attn_weight, fallback)


def entmax15_bisect(logits: torch.Tensor, dim: int = -1, n_iter: int = 20) -> torch.Tensor:
    """
    Entmax with alpha=1.5 computed by bisection.
    Produces sparse probabilities while keeping a gradient path for suppressed entries.
    """
    logits = logits - logits.max(dim=dim, keepdim=True).values
    tau_lo = 0.5 * logits.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = 0.5 * logits.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) / 2.0
        probs = torch.clamp(0.5 * logits - tau_mid, min=0.0) ** 2
        probs_sum = probs.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(probs_sum > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(probs_sum <= 1.0, tau_mid, tau_hi)

    probs = torch.clamp(0.5 * logits - tau_hi, min=0.0) ** 2
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_attention(attn_weight: torch.Tensor, normalizer: str = "softmax") -> torch.Tensor:
    if normalizer == "softmax":
        probs = torch.softmax(attn_weight, dim=-1)
    elif normalizer == "entmax15":
        probs = entmax15_bisect(attn_weight, dim=-1)
    else:
        raise ValueError(f"Unknown attention normalizer: {normalizer}")
    return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None,
                                 score_mod=None, dropout_p: float = 0.0, is_causal: bool = False, scale=None,
                                 enable_gqa: bool = False, sparsify=None, training: bool = True,
                                 normalizer: str = "softmax") -> tuple[torch.Tensor, torch.Tensor]:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_weight = attn_weight.masked_fill(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        else:
            while attn_mask.dim() < attn_weight.dim():
                attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask

    if score_mod is not None:
        attn_weight = score_mod(attn_weight)

    # sparsify the softmax attention matrix
    if sparsify is not None:
        attn_weight = _stabilize_attention_logits(attn_weight)
        attn_soft = _normalize_attention(attn_weight, normalizer=normalizer)
        avg_proba = torch.mean(attn_soft, dim=0)

        new_attn_mask = sparsify(avg_proba)
        new_attn_mask = new_attn_mask.bool()
        new_attn_mask = new_attn_mask.to(query.device)

        if new_attn_mask is not None:
            if new_attn_mask.dim() == 2:
                new_attn_mask = new_attn_mask.unsqueeze(0).unsqueeze(0)
            elif new_attn_mask.dim() == 3:
                new_attn_mask = new_attn_mask.unsqueeze(0)
            else:
                while new_attn_mask.dim() < attn_weight.dim():
                    new_attn_mask = new_attn_mask.unsqueeze(0)
            if new_attn_mask.dtype == torch.bool:
                attn_weight = attn_weight.masked_fill(new_attn_mask.logical_not(), float("-inf"))
            else:
                attn_weight = attn_weight + new_attn_mask

    attn_weight = _stabilize_attention_logits(attn_weight)
    attn_soft = _normalize_attention(attn_weight, normalizer=normalizer)
    # Respect module train/eval mode so inference stays deterministic.
    attn_soft = torch.dropout(attn_soft, dropout_p, train=training)

    return attn_soft @ value, attn_soft


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
            self,
            E_q: int,
            E_k: int,
            E_v: int,
            E_total: int,
            nheads: int,
            dropout: float = 0.0,
            bias=True,
            device=None,
            dtype=None,
            sparsify=None,
            attn_normalizer: str = "softmax",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.sparsify = sparsify
        self.attn_normalizer = attn_normalizer
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None,
                is_causal=False, sparsify=None) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_q, E_qk)
            key (torch.Tensor): key of shape (N, L_kv, E_qk)
            value (torch.Tensor): value of shape (N, L_kv, E_v)
            attn_mask (torch.Tensor, optional): attention mask of shape (N, L_q, L_kv) to pass to sdpa. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias), F.linear(value,
                                                                                                                 v_weight,
                                                                                                                 v_bias)

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output, attn_weights = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout,
            is_causal=is_causal,
            attn_mask=attn_mask,
            sparsify=sparsify,
            training=self.training,
            normalizer=self.attn_normalizer,
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        self.last_attn_weights = attn_weights.detach()

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_layer: "TransformerEncoderLayer",
            num_layers: int,
            norm: Optional[nn.Module] = None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal: bool = False,
                sparsify=None) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, attn_mask=mask, is_causal=is_causal, sparsify=sparsify)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            decoder_layer: "TransformerDecoderLayer",
            num_layers: int,
            norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_is_causal=False,
            memory_is_causal=False
    ):
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation: nn.Module = torch.nn.functional.relu,
            layer_norm_eps=1e-5,
            norm_first=True,
            bias=True,
            device=None,
            dtype=None,
            attn_normalizer: str = "softmax",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            attn_normalizer=attn_normalizer,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, attn_mask, is_causal=False, sparsify=None):
        x = self.self_attn(x, x, x, is_causal=is_causal, attn_mask=attn_mask, sparsify=sparsify)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, attn_mask=None, src_mask=None, is_causal=False, sparsify=None):
        '''
        Arguments:
            src: (batch_size, seq_len, d_model)
            src_mask: (batch_size, seq_len, seq_len)
            is_causal: bool
        '''
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal, sparsify=sparsify)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask=attn_mask, is_causal=is_causal, sparsify=sparsify))
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation: nn.Module = torch.nn.functional.relu,
            layer_norm_eps=1e-5,
            norm_first=False,
            bias=True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    # self-attention block
    def _sa_block(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
            self,
            x: torch.Tensor,
            mem: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_is_causal=False,
            memory_is_causal=False,
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x


class DiagonalGatedTemporalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with learnable gated coefficients applied to each diagonal
    of the attention matrix. Each diagonal corresponds to a different temporal lag.
    Gates are shared across heads.
    """

    def __init__(
            self,
            E_q,
            E_k,
            E_v,
            E_total,
            nheads,
            dropout=0.0,
            bias=True,
            device=None,
            dtype=None,
            max_seq_len=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        assert E_total % nheads == 0
        self.nheads = nheads
        self.d_head = E_total // nheads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(E_total, E_total, bias=bias, **factory_kwargs)

        self.dropout = nn.Dropout(dropout)

        self.max_seq_len = max_seq_len
        if max_seq_len is not None:
            self.gated_coeffs = nn.Parameter(
                # torch.ones(max_seq_len, device=device, dtype=dtype) * 1.0
                # torch.tensor([1.0 * np.exp(-0.1*k) for k in range(max_seq_len)]
                torch.tensor([1.0 for k in range(max_seq_len)]
                             # torch.tensor(0.*torch.randn(max_seq_len) + 1
                             # torch.tensor([np.random.uniform() for k in range(max_seq_len)]
                             , dtype=torch.float32)
            )
        else:
            self.gated_coeffs = None

    def _init_gated_coeffs(self, T, device, dtype):
        self.max_seq_len = T
        gated_coeffs_tensor = torch.ones(T, device=device, dtype=dtype) * 1.0
        gated_coeffs_tensor = torch.tensor([1.0 * np.exp(-k) for k in range(T)], dtype=torch.float32)
        self.register_parameter('gated_coeffs', nn.Parameter(gated_coeffs_tensor))

    def _apply_diagonal_gates(self, attn_weights, gated_coeffs):

        B, H, T, T_sq = attn_weights.shape
        assert T == T_sq, "Attention matrix must be square"
        assert gated_coeffs.shape[0] == T, f"Expected {T} coefficients, got {gated_coeffs.shape[0]}"

        device = attn_weights.device
        dtype = attn_weights.dtype

        row_indices = torch.arange(T, device=device).unsqueeze(1).expand(T, T)
        col_indices = torch.arange(T, device=device).unsqueeze(0).expand(T, T)
        lag_indices = row_indices - col_indices

        valid_mask = (lag_indices >= 0) & (lag_indices < T)

        gate_matrix = torch.ones(T, T, device=device, dtype=dtype)
        gate_matrix[valid_mask] = gated_coeffs[lag_indices[valid_mask]]

        gate_matrix_expanded = gate_matrix[None, None, :, :]
        gated_weights = attn_weights * gate_matrix_expanded

        # print('gated coef', gated_coeffs)

        return gated_weights

    def forward(self, q, k, v, attn_mask=None, is_causal=False):
        B, T, _ = q.shape
        device = q.device
        dtype = q.dtype

        if self.gated_coeffs is None:
            self._init_gated_coeffs(T, device, dtype)
        elif T > self.max_seq_len:
            old_coeffs = self.gated_coeffs
            self._init_gated_coeffs(T, device, dtype)
            self.gated_coeffs.data[:old_coeffs.shape[0]] = old_coeffs.data

        if T <= self.gated_coeffs.shape[0]:
            gated_coeffs_used = self.gated_coeffs[:T]
        else:
            gated_coeffs_used = self.gated_coeffs

        zeros = (gated_coeffs_used > torch.mean(gated_coeffs_used) * 0.85)
        gated_coeffs_used = gated_coeffs_used * zeros
        gated_coeffs_used = gated_coeffs_used / (gated_coeffs_used.sum())

        Q = self.q_proj(q).view(B, T, self.nheads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        K = self.k_proj(k).view(B, T, self.nheads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        V = self.v_proj(v).view(B, T, self.nheads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if is_causal:
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]

            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        # softmax to get attention weights
        attn_weights = torch.softmax(self._apply_diagonal_gates(scores, gated_coeffs_used), dim=-1)  # (B, H, T, T)

        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        out = self.out_proj(out)

        return out, gated_coeffs_used


class DiagonalGatedTemporalEncoderLayer(nn.Module):
    """
    Transformer encoder layer using diagonal-gated temporal attention.
    Returns both the output and gated coefficients for regularization.
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation=torch.nn.functional.relu,
            layer_norm_eps=1e-5,
            norm_first=True,
            bias=True,
            device=None,
            dtype=None,
            max_seq_len=None,  # Add this parameter
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = DiagonalGatedTemporalMultiHeadAttention(
            d_model, d_model, d_model, d_model, nhead, dropout, bias,
            max_seq_len=max_seq_len,  # Pass it through
            **factory_kwargs
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, attn_mask=None, is_causal=False):

        attn_out, gated_coeffs = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        return self.dropout1(attn_out), gated_coeffs

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, attn_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            attn_out, gated_coeffs = self._sa_block(
                self.norm1(x), attn_mask=attn_mask, is_causal=is_causal
            )
            x = x + attn_out
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_out, gated_coeffs = self._sa_block(x, attn_mask=attn_mask, is_causal=is_causal)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self._ff_block(x))

        return x, gated_coeffs
