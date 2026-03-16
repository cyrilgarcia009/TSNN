from torch import nn
from tsnn.tstorch import transformers
import torch
import copy

"""
File contains four models:

i) GlobalMLP: Trivial MLP model without any structure, flattens N,F,T into input_dim = n_rolling * n_ts * n_f then applies an MLP.

ii) BiDimensionalMLP: Applies succesively an MLP along a first direction then an MLP along the second. 
The parameter first_direction specifies which dimension to treat first.

iii) OneDimensionalTransformer: Applies num_layers of attention in one direction specified by the parameter attn_direction. 
The other direction is compressed at the start using either a simple linear layer or an MLP.

iv) CustomBiDimensionalTransformer: Most general model to apply attention layers in both directions. 
Succesive layers are specified by for instance layers="TCTC", where "T" and "C" represent respectively time-series and cross-sectional attention.
"""


class GlobalMLP(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, F)
        """
        B = x.shape[0]
        x = x.reshape(B, -1)  # (B, T*N*F)
        out = self.network(x)  # (B, N)
        return out


class BiDimensionalMLP(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # x: (batch, n_rolling, n_ts, n_f)
        B, n_rolling, n_ts, n_f = x.shape

        if self.attn_direction == "T":

            x = x.reshape(B, n_rolling, n_ts * n_f)

            x = self.input_proj(x) + self.pos_emb_time[:, :n_rolling, :]

            x = self.encoder(x, mask=self.mask, sparsify=self.sparsify)

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
                        d_model, nhead, dim_feedforward, dropout
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
                        d_model, nhead, dim_feedforward, dropout
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

    def temporal_attention(self, x: torch.Tensor, attn_encoder) -> torch.Tensor:
        """Temporal attention: attend over the T time dimension independently for each series"""
        B, T, N, D = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        out = attn_encoder(x_flat, mask=self.mask, sparsify=self.sparsify)
        out = out.view(B, N, T, D).transpose(1, 2)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                x_att = self.temporal_attention(x, block['temporal'])
                x = block['norm'](res + self.dropout(x_att).contiguous())

            elif 'series' in block:
                # Cross-sectional block
                x_att = self.series_attention(x, block['series'])
                x = block['norm'](res + self.dropout(x_att).contiguous())

        if not self.roll_y:
            x = x[:, -1, :, :]  # Take last time step for forecasting

        return self.output_head(x).squeeze(-1)


class CustomBiDimensionalTransformerSparse(nn.Module):
    """
    CustomBiDimensionalTransformer with L1-sparse diagonal-gated temporal attention.
    Uses diagonal-gated coefficients for temporal (T) direction with L1 penalization.
    """

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

    def temporal_attention(self, x: torch.Tensor, attn_encoder) -> torch.Tensor:
        B, T, N, D = x.shape
        # Attend over time independently for each series.
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        out = attn_encoder(x_flat, mask=self.mask, sparsify=self.sparsify)
        out = out.view(B, N, T, D).transpose(1, 2)
        return out

    def temporal_attention_l1_gated(self, x: torch.Tensor, attn_layer) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, N, D = x.shape
        # Temporal sparse block also returns the gated lag coefficients.
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        out, gated_coeffs = attn_layer(x_flat, attn_mask=self.mask, is_causal=True)
        out = out.view(B, N, T, D).transpose(1, 2)
        return out, gated_coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, N, _ = x.shape
        x = self.input_proj(x)

        x = x + self.pos_emb_time[:, :T, None, :] + self.pos_emb_series[:, None, :N, :]
        self._last_gated_coeffs = []

        for block in self.blocks:
            res = x

            if 'temporal_l1_gated' in block:
                x_att, gated_coeffs = self.temporal_attention_l1_gated(x, block['temporal_l1_gated'])
                x = block['norm'](res + self.dropout(x_att).contiguous())
                self._last_gated_coeffs.append(gated_coeffs)

            elif 'temporal' in block:
                x_att = self.temporal_attention(x, block['temporal'])
                x = block['norm'](res + self.dropout(x_att).contiguous())

            elif 'series' in block:
                x_att = self.series_attention(x, block['series'])
                x = block['norm'](res + self.dropout(x_att).contiguous())

        if not self.roll_y:
            x = x[:, -1, :, :]

        return self.output_head(x).squeeze(-1)

    def get_gated_coeffs(self) -> list[torch.Tensor]:
        return self._last_gated_coeffs
