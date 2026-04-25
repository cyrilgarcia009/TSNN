from torch import nn
from tsnn.tstorch import transformers
import torch
import copy


def _build_temporal_attn_mask(base_mask, pad_mask: torch.Tensor, repeat_factor: int = 1) -> torch.Tensor:
    """
    Combines the causal mask with a batch padding mask.
    Invalid query rows fall back to the identity so attention stays finite.
    """
    B, T = pad_mask.shape
    device = pad_mask.device

    if base_mask is None:
        attn_mask = torch.ones(B, T, T, dtype=torch.bool, device=device)
    else:
        attn_mask = base_mask.to(device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1).clone()

    key_mask = pad_mask[:, None, :].expand(B, T, T)
    query_mask = pad_mask[:, :, None].expand(B, T, T)
    attn_mask = attn_mask & key_mask

    eye = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)
    attn_mask = torch.where(query_mask, attn_mask, eye)

    if repeat_factor > 1:
        attn_mask = attn_mask.repeat_interleave(repeat_factor, dim=0)

    return attn_mask


def _build_joint_time_mask(n_rolling: int, n_ts: int, device=None) -> torch.Tensor:
    """Causal mask on the flattened (time, series) grid."""
    time_index = torch.arange(n_rolling, device=device).repeat_interleave(n_ts)
    return time_index[:, None] >= time_index[None, :]


def _build_spatiotemporal_attn_mask(base_mask: torch.Tensor, pad_mask: torch.Tensor, n_ts: int) -> torch.Tensor:
    """
    Extends the temporal padding mask to the flattened (time, series) grid.
    Invalid query rows fall back to the identity so attention stays finite.
    """
    B, T = pad_mask.shape
    device = pad_mask.device
    L = T * n_ts

    attn_mask = base_mask.to(device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1).clone()
    token_mask = pad_mask.repeat_interleave(n_ts, dim=1)

    key_mask = token_mask[:, None, :].expand(B, L, L)
    query_mask = token_mask[:, :, None].expand(B, L, L)
    attn_mask = attn_mask & key_mask

    eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    attn_mask = torch.where(query_mask, attn_mask, eye)

    return attn_mask


class GlobalMLP(nn.Module):
    """Plain MLP baseline on the full flattened rolling window."""

    def __init__(
            self,
            n_ts,  # N
            n_f,  # F
            n_rolling,  # T
            hidden_dim=512,
            num_layers=4,
            dropout=0.1,
    ):
        super().__init__()

        input_dim = n_rolling * n_ts * n_f

        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        # Final projection to N targets
        layers.append(nn.Linear(hidden_dim, n_ts))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, N, F)
        """
        B = x.shape[0]
        x = x.reshape(B, -1)  # (B, T*N*F)
        out = self.network(x)  # (B, N)
        return out


class GlobalLSTM(nn.Module):
    """Global LSTM over flattened panel features across time, predicting all series."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
    ):
        super().__init__()
        self.n_ts = n_ts
        input_dim = n_ts * n_f
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, n_ts)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, N, F)
        """
        B, T, N, F = x.shape
        x = x.reshape(B, T, N * F)

        if pad_mask is not None:
            lengths = pad_mask.sum(dim=1).clamp(min=1).to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.encoder(packed)
            last_hidden = hidden[-1]
        else:
            _, (hidden, _) = self.encoder(x)
            last_hidden = hidden[-1]

        return self.output(self.dropout(last_hidden))


class BiDimensionalMLP(nn.Module):
    """Two-stage MLP baseline that processes one axis and then the other."""

    def __init__(
            self,
            n_ts,  # N
            n_f,  # F
            n_rolling,  # T
            first_direction="T",  # Must be "T" or "C" for time-series or cross-sectional
            hidden_dim_mlp1=256,
            hidden_dim_mlp2=512,
            num_layers_mlp1=3,
            num_layers_mlp2=3,
            dropout=0.1,

    ):
        super().__init__()

        self.n_ts = n_ts
        self.hidden_dim_mlp1 = hidden_dim_mlp1
        self.first_direction = first_direction

        if self.first_direction == "T":
            self.input_dim1 = n_rolling * n_f
            self.input_dim2 = n_ts * hidden_dim_mlp1
        else:
            self.input_dim1 = n_ts * n_f
            self.input_dim2 = n_rolling * hidden_dim_mlp1

        mlp1_layers = []
        dim = self.input_dim1
        for _ in range(num_layers_mlp1):
            mlp1_layers.extend([
                nn.Linear(dim, hidden_dim_mlp1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim_mlp1)
            ])
            dim = hidden_dim_mlp1
        # Final local projection
        mlp1_layers.append(nn.Linear(dim, hidden_dim_mlp1))
        self.mlp1 = nn.Sequential(*mlp1_layers)

        mlp2_layers = []
        dim = self.input_dim2
        for _ in range(num_layers_mlp2):
            mlp2_layers.extend([
                nn.Linear(dim, hidden_dim_mlp2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim_mlp2)
            ])
            dim = hidden_dim_mlp2
        mlp2_layers.append(nn.Linear(hidden_dim_mlp2, n_ts))
        self.mlp2 = nn.Sequential(*mlp2_layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, N, F)
        """
        B, T, N, F = x.shape

        if self.first_direction == "T":

            x = x.permute(0, 2, 1, 3).reshape(B * N, T * F)  # (B*N, T*F)

            local_emb = self.mlp1(x)  # (B*N, hidden_dim_mlp1)

            x_global = local_emb.view(B, N * self.hidden_dim_mlp1)  # (B, N*hidden_dim_mlp1)

            out = self.mlp2(x_global)  # (B, N)

        else:

            x = x.reshape(B * T, N * F)  # (B*T, N*F)

            local_emb = self.mlp1(x)  # (B*T, hidden_dim_mlp1)

            x_global = local_emb.view(B, T * self.hidden_dim_mlp1)  # (B, T*hidden_dim_mlp1)

            out = self.mlp2(x_global)  # (B, N)

        return out


class OneDimensionalTransformer(nn.Module):
    """Transformer applied along a single axis after compressing the other axis."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            mask,
            attn_direction="T",  # Should be "T" or "C"
            compression="SimpleLin",  # Should be "SimpleLin" or "MLP"
            d_model=128,
            num_mlp_layers=4,  # Only used when compression = "MLP"
            nhead=8,
            num_attn_layers=2,
            dim_feedforward=512,
            dropout=0.2,
            sparsify=None,
            roll_y=False
    ):
        super().__init__()
        self.attn_direction = attn_direction  # Should be "T" or "C" for time series or cross-sectional
        self.compression = compression
        self.mask = mask
        self.sparsify = sparsify
        self.d_model = d_model
        self.num_layers_mlp = num_mlp_layers
        self.roll_y = roll_y

        if attn_direction == "T":
            self.input_dim = n_ts * n_f
        else:
            self.input_dim = n_rolling * n_f

        if self.compression == "SimpleLin":
            self.input_proj = nn.Linear(self.input_dim, d_model)

        else:
            mlp_layers = []
            dim = self.input_dim
            for _ in range(num_mlp_layers):
                mlp_layers.extend([
                    nn.Linear(dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model)
                ])
                dim = d_model
            # Final local projection
            mlp_layers.append(nn.Linear(dim, d_model))
            self.input_proj = nn.Sequential(*mlp_layers)

        # Positional encoding (learned or sinusoidal)
        self.pos_emb_time = nn.Parameter(torch.randn(1, n_rolling, d_model))
        self.pos_emb_series = nn.Parameter(torch.randn(1, n_ts, d_model))

        # Transformer encoder
        encoder_layer = transformers.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = transformers.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

        if self.attn_direction == "T":
            self.output_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_ts)
            )
        else:
            self.output_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )

    def forward(self, x: torch.Tensor, mask=None, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, n_rolling, n_ts, n_f)
        B, n_rolling, n_ts, n_f = x.shape

        if self.attn_direction == "T":

            x = x.reshape(B, n_rolling, n_ts * n_f)

            x = self.input_proj(x) + self.pos_emb_time[:, :n_rolling, :]

            attn_mask = self.mask
            if pad_mask is not None:
                attn_mask = _build_temporal_attn_mask(self.mask, pad_mask)

            x = self.encoder(x, mask=attn_mask, sparsify=self.sparsify)

            if self.roll_y == True:
                return self.output_head(x)

            else:
                return self.output_head(x[:, -1, :])


        else:

            x = x.transpose(1, 2).reshape(B, n_ts, n_rolling * n_f)

            x = self.input_proj(x) + self.pos_emb_series[:, :n_ts, :]

            x = self.encoder(x, sparsify=self.sparsify)

            return self.output_head(x).squeeze(-1)


class CustomBiDimensionalTransformer(nn.Module):
    """Alternating temporal and cross-sectional transformer blocks, e.g. TCTC."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            mask,
            d_model=128,
            nhead=8,
            layers="TCTC",
            dim_feedforward=512,
            dropout=0.2,
            sparsify=None,
            roll_y=False,
            embeddings="both",
            attn_normalizer="softmax",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_rolling = n_rolling
        self.n_ts = n_ts
        self.mask = mask
        self.sparsify = sparsify
        self.roll_y = roll_y
        self.layers = layers

        self.input_proj = nn.Linear(n_f, d_model)

        # Broadcasted positional embeddings
        self.pos_emb_time = nn.Parameter(torch.randn(1, n_rolling, d_model))
        self.pos_emb_series = nn.Parameter(torch.randn(1, n_ts, d_model))
        self.embeddings = embeddings

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for symbol in layers:
            if symbol == 'T':
                temporal_encoder = transformers.TransformerEncoder(
                    transformers.TransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout, attn_normalizer=attn_normalizer
                    ),
                    num_layers=1
                )
                self.blocks.append(nn.ModuleDict({
                    'temporal': temporal_encoder,
                    'norm': nn.LayerNorm(d_model),
                }))
            elif symbol == 'C':
                series_encoder = transformers.TransformerEncoder(
                    transformers.TransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout, attn_normalizer=attn_normalizer
                    ),
                    num_layers=1
                )
                self.blocks.append(nn.ModuleDict({
                    'series': series_encoder,
                    'norm': nn.LayerNorm(d_model),
                }))
            else:
                raise ValueError(f"Invalid character in layers string: '{symbol}'. Only 'T' and 'C' are allowed.")

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def series_attention(self, x: torch.Tensor, attn_encoder) -> torch.Tensor:
        """Cross-sectional attention: attend over the N series dimension independently for each (B,T)"""
        B, T, N, D = x.shape
        x_flat = x.view(B * T, N, D)
        out = attn_encoder(x_flat, sparsify=self.sparsify)
        out = out.view(B, T, N, D)
        return out

    def temporal_attention(self, x: torch.Tensor, attn_encoder, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """Temporal attention: attend over the T time dimension independently for each series"""
        B, T, N, D = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        attn_mask = self.mask
        if pad_mask is not None:
            attn_mask = _build_temporal_attn_mask(self.mask, pad_mask, repeat_factor=N)
        out = attn_encoder(x_flat, mask=attn_mask, sparsify=self.sparsify)
        out = out.view(B, N, T, D).transpose(1, 2)
        return out

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, N, F)
        B, T, N, _ = x.shape
        x = self.input_proj(x)

        # Add learnable positional embeddings (broadcasted over B, T, N)
        if self.embeddings=="both":
            x = x + self.pos_emb_time[:, :T, None, :]  + self.pos_emb_series[:, None, :N, :]
        elif self.embeddings=="C":
            x = x + self.pos_emb_series[:, None, :N, :]
        elif self.embeddings=="T":
            x = x + self.pos_emb_time[:, :T, None, :]

        # x = self.dropout(x)

        # Process each block
        for block in self.blocks:
            res = x

            if 'temporal' in block:
                # Temporal block
                x_att = self.temporal_attention(x, block['temporal'], pad_mask=pad_mask)
                x = block['norm'](res + self.dropout(x_att).contiguous())

            elif 'series' in block:
                # Cross-sectional block
                x_att = self.series_attention(x, block['series'])
                x = block['norm'](res + self.dropout(x_att).contiguous())

        if not self.roll_y:
            x = x[:, -1, :, :]  # Take last time step for forecasting

        return self.output_head(x).squeeze(-1)


class MinimalDoubleShiftTransformer(nn.Module):
    """One temporal block per series, then one cross-sectional block at the final time."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            roll_y=False,
    ):
        super().__init__()
        self.n_ts = n_ts
        self.n_f = n_f
        self.n_rolling = n_rolling
        self.roll_y = roll_y
        self.d_model = d_model

        self.input_proj = nn.Linear(n_f, d_model)
        self.pos_emb_time = nn.Parameter(torch.randn(1, n_rolling, d_model) * 0.02)
        self.pos_emb_series = nn.Parameter(torch.randn(1, n_ts, d_model) * 0.02)

        self.temporal_block = transformers.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=torch.nn.functional.gelu,
            norm_first=False,
        )
        self.cross_block = transformers.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=torch.nn.functional.gelu,
            norm_first=False,
        )
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, N, F)
        B, T, N, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_emb_time[:, :T, None, :] + self.pos_emb_series[:, None, :N, :]

        h_temp = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        temporal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        if pad_mask is not None:
            temporal_mask = _build_temporal_attn_mask(temporal_mask, pad_mask, repeat_factor=N)
        h_temp = self.temporal_block(h_temp, attn_mask=temporal_mask)
        h_temp = h_temp.view(B, N, T, self.d_model).permute(0, 2, 1, 3).contiguous()

        if self.roll_y:
            cross_input = h_temp.view(B * T, N, self.d_model)
            cross_out = self.cross_block(cross_input)
            cross_out = cross_out.view(B, T, N, self.d_model)
            return self.output_head(cross_out).squeeze(-1)

        h_last = h_temp[:, -1, :, :]
        h_cross = self.cross_block(h_last)
        return self.output_head(h_cross).squeeze(-1)


class JointSpatioTemporalTransformer(nn.Module):
    """Attention on the flattened (time, series) grid in a single joint block."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.2,
            sparsify=None,
            roll_y=False,
            embeddings="both",
    ):
        super().__init__()
        self.n_ts = n_ts
        self.n_rolling = n_rolling
        self.roll_y = roll_y
        self.sparsify = sparsify
        self.embeddings = embeddings

        self.input_proj = nn.Linear(n_f, d_model)
        self.pos_emb_time = nn.Parameter(torch.randn(1, n_rolling, d_model))
        self.pos_emb_series = nn.Parameter(torch.randn(1, n_ts, d_model))

        encoder_layer = transformers.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = transformers.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.register_buffer(
            "joint_time_mask",
            _build_joint_time_mask(n_rolling, n_ts),
            persistent=False,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, N, F)
        B, T, N, _ = x.shape
        x = self.input_proj(x)

        if self.embeddings == "both":
            x = x + self.pos_emb_time[:, :T, None, :] + self.pos_emb_series[:, None, :N, :]
        elif self.embeddings == "C":
            x = x + self.pos_emb_series[:, None, :N, :]
        elif self.embeddings == "T":
            x = x + self.pos_emb_time[:, :T, None, :]

        # Flatten the (time, series) grid so attention can target a lag/source-series pair directly.
        x = x.reshape(B, T * N, -1)

        attn_mask = self.joint_time_mask
        if pad_mask is not None:
            attn_mask = _build_spatiotemporal_attn_mask(self.joint_time_mask, pad_mask, N)

        x = self.encoder(x, mask=attn_mask, sparsify=self.sparsify)
        x = x.view(B, T, N, -1)

        if not self.roll_y:
            x = x[:, -1, :, :]

        return self.output_head(x).squeeze(-1)


class FeaturewiseStaticRoutingBiDimensionalModel(nn.Module):
    """Featurewise static lag and series routing before late feature aggregation."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            d_model=64,
            dropout=0.1,
            roll_y=False,
            temperature=0.1,
    ):
        super().__init__()
        self.n_ts = n_ts
        self.n_f = n_f
        self.n_rolling = n_rolling
        self.roll_y = roll_y
        self.temperature = temperature

        self.dropout = nn.Dropout(dropout)

        # Each feature gets its own lag kernel so feature-specific shifts can coexist.
        self.lag_logits = nn.Parameter(torch.zeros(n_f, n_rolling))
        self.series_logits = nn.Parameter(torch.zeros(n_f, n_ts, n_ts))
        self.register_buffer(
            "time_causal_mask",
            torch.tril(torch.ones(n_rolling, n_rolling, dtype=torch.bool)),
            persistent=False,
        )

        self.output_head = nn.Sequential(
            # Keep raw feature amplitudes available to the aggregator.
            nn.Linear(n_f, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def _time_weights(self, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Build one stationary causal lag kernel per feature.
        This lets one feature prefer lag 1 while another prefers lag 10.
        """
        lag_scores = torch.softmax(self.lag_logits[:, :self.n_rolling] / self.temperature, dim=-1)
        T = pad_mask.shape[1] if pad_mask is not None else self.n_rolling
        lag_scores = lag_scores[:, :T]

        time_idx = torch.arange(T, device=lag_scores.device)
        lag_idx = time_idx[:, None] - time_idx[None, :]

        base_weights = torch.zeros(self.n_f, T, T, device=lag_scores.device, dtype=lag_scores.dtype)
        valid = lag_idx >= 0
        base_weights[:, valid] = lag_scores[:, lag_idx[valid]]
        base_weights = base_weights / base_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if pad_mask is None:
            return base_weights

        B = pad_mask.shape[0]
        valid_mask = _build_temporal_attn_mask(self.time_causal_mask[:T, :T], pad_mask)
        weights = base_weights.unsqueeze(0).expand(B, -1, -1, -1).clone()
        weights = weights * valid_mask[:, None, :, :].to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return weights

    def _series_weights(self) -> torch.Tensor:
        logits = self.series_logits / self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, N, F)
        time_weights = self._time_weights(pad_mask=pad_mask)
        if time_weights.dim() == 3:
            x = torch.einsum('fts,bsnf->btnf', time_weights, x)
        else:
            x = torch.einsum('bfts,bsnf->btnf', time_weights, x)

        series_weights = self._series_weights()
        x = torch.einsum('fnm,btmf->btnf', series_weights, x)
        x = self.dropout(x)

        if not self.roll_y:
            x = x[:, -1, :, :]

        return self.output_head(x).squeeze(-1)


class RoutingBiasedBiDimensionalTransformer(nn.Module):
    """T/C transformer with learned static routing biases added to attention scores."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            mask,
            d_model=128,
            nhead=8,
            layers="TCTC",
            dim_feedforward=512,
            dropout=0.2,
            sparsify=None,
            roll_y=False,
            embeddings="both",
            temperature=1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_ts = n_ts
        self.n_rolling = n_rolling
        self.mask = mask
        self.sparsify = sparsify
        self.roll_y = roll_y
        self.layers = layers
        self.embeddings = embeddings
        self.temperature = temperature

        self.input_proj = nn.Linear(n_f, d_model)
        self.pos_emb_time = nn.Parameter(torch.randn(1, n_rolling, d_model))
        self.pos_emb_series = nn.Parameter(torch.randn(1, n_ts, d_model))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        self.temporal_biases = nn.ParameterList()
        self.series_biases = nn.ParameterList()
        for symbol in layers:
            if symbol == 'T':
                temporal_encoder = transformers.TransformerEncoder(
                    transformers.TransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout
                    ),
                    num_layers=1
                )
                self.blocks.append(nn.ModuleDict({
                    'temporal': temporal_encoder,
                    'norm': nn.LayerNorm(d_model),
                }))
                self.temporal_biases.append(nn.Parameter(torch.zeros(n_rolling)))
            elif symbol == 'C':
                series_encoder = transformers.TransformerEncoder(
                    transformers.TransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout
                    ),
                    num_layers=1
                )
                self.blocks.append(nn.ModuleDict({
                    'series': series_encoder,
                    'norm': nn.LayerNorm(d_model),
                }))
                self.series_biases.append(nn.Parameter(torch.zeros(n_ts, n_ts)))
            else:
                raise ValueError(f"Invalid character in layers string: '{symbol}'. Only 'T' and 'C' are allowed.")

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def _temporal_bias_mask(self, T: int, pad_mask: torch.Tensor = None, bias_index: int = 0) -> torch.Tensor:
        lag_logits = self.temporal_biases[bias_index][:T] / self.temperature
        time_idx = torch.arange(T, device=lag_logits.device)
        lag_idx = time_idx[:, None] - time_idx[None, :]

        attn_bias = torch.full((T, T), float("-inf"), device=lag_logits.device)
        valid = lag_idx >= 0
        attn_bias[valid] = lag_logits[lag_idx[valid]]

        if pad_mask is None:
            return attn_bias

        B = pad_mask.shape[0]
        attn_bias = attn_bias.unsqueeze(0).expand(B, -1, -1).clone()
        valid_mask = _build_temporal_attn_mask(self.mask, pad_mask)
        attn_bias = attn_bias.masked_fill(valid_mask.logical_not(), float("-inf"))
        return attn_bias

    def _series_bias_mask(self, N: int, bias_index: int = 0) -> torch.Tensor:
        return self.series_biases[bias_index][:N, :N] / self.temperature

    def series_attention(self, x: torch.Tensor, attn_encoder, bias_index: int = 0) -> torch.Tensor:
        B, T, N, D = x.shape
        x_flat = x.view(B * T, N, D)
        attn_mask = self._series_bias_mask(N, bias_index=bias_index)
        out = attn_encoder(x_flat, mask=attn_mask, sparsify=self.sparsify)
        return out.view(B, T, N, D)

    def temporal_attention(self, x: torch.Tensor, attn_encoder, pad_mask: torch.Tensor = None, bias_index: int = 0) -> torch.Tensor:
        B, T, N, D = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        attn_mask = self._temporal_bias_mask(T, pad_mask=pad_mask, bias_index=bias_index)
        if pad_mask is not None:
            attn_mask = attn_mask.repeat_interleave(N, dim=0)
        out = attn_encoder(x_flat, mask=attn_mask, sparsify=self.sparsify)
        return out.view(B, N, T, D).transpose(1, 2)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        x = self.input_proj(x)

        if self.embeddings == "both":
            x = x + self.pos_emb_time[:, :T, None, :] + self.pos_emb_series[:, None, :N, :]
        elif self.embeddings == "C":
            x = x + self.pos_emb_series[:, None, :N, :]
        elif self.embeddings == "T":
            x = x + self.pos_emb_time[:, :T, None, :]

        temporal_bias_idx = 0
        series_bias_idx = 0
        for block in self.blocks:
            if 'temporal' in block:
                res = x
                x_att = self.temporal_attention(x, block['temporal'], pad_mask=pad_mask, bias_index=temporal_bias_idx)
                x = block['norm'](res + self.dropout(x_att).contiguous())
                temporal_bias_idx += 1
            elif 'series' in block:
                res = x
                x_att = self.series_attention(x, block['series'], bias_index=series_bias_idx)
                x = block['norm'](res + self.dropout(x_att).contiguous())
                series_bias_idx += 1

        if not self.roll_y:
            x = x[:, -1, :, :]

        return self.output_head(x).squeeze(-1)


class RelativeTemporalAbsoluteCSBiDimensionalTransformer(RoutingBiasedBiDimensionalTransformer):
    """Same TCTC stack, but time is purely relative while CS keeps absolute identity plus B."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            mask,
            d_model=128,
            nhead=8,
            layers="TCTC",
            dim_feedforward=512,
            dropout=0.2,
            sparsify=None,
            roll_y=False,
            temperature=1.0,
            use_series_embeddings=True,
    ):
        super().__init__(
            n_ts=n_ts,
            n_f=n_f,
            n_rolling=n_rolling,
            mask=mask,
            d_model=d_model,
            nhead=nhead,
            layers=layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            sparsify=sparsify,
            roll_y=roll_y,
            embeddings="C",
            temperature=temperature,
        )
        self.use_series_embeddings = use_series_embeddings

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        x = self.input_proj(x)

        # Temporal structure comes only from relative lag biases; series keep absolute identities.
        if self.use_series_embeddings:
            x = x + self.pos_emb_series[:, None, :N, :]

        temporal_bias_idx = 0
        series_bias_idx = 0
        for block in self.blocks:
            if 'temporal' in block:
                res = x
                x_att = self.temporal_attention(x, block['temporal'], pad_mask=pad_mask, bias_index=temporal_bias_idx)
                x = block['norm'](res + self.dropout(x_att).contiguous())
                temporal_bias_idx += 1
            elif 'series' in block:
                res = x
                x_att = self.series_attention(x, block['series'], bias_index=series_bias_idx)
                x = block['norm'](res + self.dropout(x_att).contiguous())
                series_bias_idx += 1

        if not self.roll_y:
            x = x[:, -1, :, :]

        return self.output_head(x).squeeze(-1)


class CustomBiDimensionalTransformerSparse(nn.Module):
    """T/C transformer with diagonal-gated sparse temporal attention."""

    def __init__(
            self,
            n_ts,
            n_f,
            n_rolling,
            mask,
            d_model=128,
            nhead=8,
            layers="TCTC",
            dim_feedforward=512,
            dropout=0.2,
            sparsify=None,
            roll_y=False,
            lambda_l1=0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.n_rolling = n_rolling
        self.n_ts = n_ts
        self.mask = mask
        self.sparsify = sparsify
        self.roll_y = roll_y
        self.layers = layers
        self.lambda_l1 = lambda_l1

        self.input_proj = nn.Linear(n_f, d_model)

        self.pos_emb_time = nn.Parameter(torch.randn(1, n_rolling, d_model))
        self.pos_emb_series = nn.Parameter(torch.randn(1, n_ts, d_model))

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for symbol in layers:
            if symbol == 'T':
                temporal_layer = transformers.DiagonalGatedTemporalEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout,
                    max_seq_len=n_rolling
                )

                self.blocks.append(nn.ModuleDict({
                    'temporal_l1_gated': temporal_layer,
                    'norm': nn.LayerNorm(d_model),
                }))
            elif symbol == 'C':
                series_layer = transformers.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout
                )
                self.blocks.append(nn.ModuleDict({
                    'series': series_layer,
                    'norm': nn.LayerNorm(d_model),
                }))
            else:
                raise ValueError(f"Invalid character in layers string: '{symbol}'. Only 'T' and 'C' are allowed.")

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self._last_gated_coeffs = []

    def series_attention(self, x: torch.Tensor, attn_layer) -> torch.Tensor:
        B, T, N, D = x.shape
        # Attend over the series axis independently at each time step.
        x_flat = x.view(B * T, N, D)
        out = attn_layer(x_flat, sparsify=self.sparsify)
        out = out.view(B, T, N, D)
        return out

    def temporal_attention(self, x: torch.Tensor, attn_encoder, pad_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, N, D = x.shape
        # Attend over time independently for each series.
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        attn_mask = self.mask
        if pad_mask is not None:
            attn_mask = _build_temporal_attn_mask(self.mask, pad_mask, repeat_factor=N)
        out = attn_encoder(x_flat, mask=attn_mask, sparsify=self.sparsify)
        out = out.view(B, N, T, D).transpose(1, 2)
        return out

    def temporal_attention_l1_gated(self, x: torch.Tensor, attn_layer, pad_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, N, D = x.shape
        # Temporal sparse block also returns the gated lag coefficients.
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        attn_mask = self.mask
        if pad_mask is not None:
            attn_mask = _build_temporal_attn_mask(self.mask, pad_mask, repeat_factor=N)
        out, gated_coeffs = attn_layer(x_flat, attn_mask=attn_mask, is_causal=False)
        out = out.view(B, N, T, D).transpose(1, 2)
        return out, gated_coeffs

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:

        B, T, N, _ = x.shape
        x = self.input_proj(x)

        x = x + self.pos_emb_time[:, :T, None, :] + self.pos_emb_series[:, None, :N, :]
        self._last_gated_coeffs = []

        for block in self.blocks:
            res = x

            if 'temporal_l1_gated' in block:
                x_att, gated_coeffs = self.temporal_attention_l1_gated(x, block['temporal_l1_gated'], pad_mask=pad_mask)
                x = block['norm'](res + self.dropout(x_att).contiguous())
                self._last_gated_coeffs.append(gated_coeffs)

            elif 'temporal' in block:
                x_att = self.temporal_attention(x, block['temporal'], pad_mask=pad_mask)
                x = block['norm'](res + self.dropout(x_att).contiguous())

            elif 'series' in block:
                x_att = self.series_attention(x, block['series'])
                x = block['norm'](res + self.dropout(x_att).contiguous())

        if not self.roll_y:
            x = x[:, -1, :, :]

        return self.output_head(x).squeeze(-1)

    def get_gated_coeffs(self) -> list[torch.Tensor]:
        return self._last_gated_coeffs
