from torch import nn
from tsnn.tstorch import transformers
import torch

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

    def forward(self, x):
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

        local_layers = []
        dim = n_rolling * n_f
        for i in range(num_layers_mlp1):
            local_layers.extend([
                nn.Linear(dim, hidden_dim_mlp1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim_mlp1)
            ])
            dim = hidden_dim_mlp1
        # Final local projection
        local_layers.append(nn.Linear(dim, hidden_dim_mlp1))
        self.mlp1 = nn.Sequential(*local_layers)

        global_layers = []
        dim = n_ts * hidden_dim_mlp1
        for i in range(num_layers_mlp2):
            global_layers.extend([
                nn.Linear(dim, hidden_dim_mlp2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim_mlp2)
            ])
            dim = hidden_dim_mlp2
        global_layers.append(nn.Linear(hidden_dim_mlp2, n_ts))
        self.mlp2 = nn.Sequential(*global_layers)

    def forward(self, x):
        """
        x: (B, T, N, F)
        """
        B, T, N, F = x.shape

        if self.first_direction == "T":

            # Reshape so each series becomes a separate "sample"
            x = x.permute(0, 2, 1, 3).reshape(B * N, T * F)  # (B*N, T*F)

            # Local processing: each series gets its own embedding
            local_emb = self.mlp1(x)  # (B*N, hidden_dim_mlp1)

            # Reshape back to have the series dimension explicit
            x_global = local_emb.view(B, N * self.hidden_dim_mlp1)  # (B, N*hidden_dim_mlp1)

            # Global mixing across series
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
        self.pos_emb_time = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb_series = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = transformers.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = transformers.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, n_rolling, n_ts, n_f)
        B, n_rolling, n_ts, n_f = x.shape

        if self.attn_direction == "T":

            x = x.reshape(B, n_rolling, n_ts * n_f)

            x = self.input_proj(x) + self.pos_emb_time

            x = self.encoder(x, mask=self.mask, sparsify=self.sparsify)

        else:

            x = x.transpose(1, 2).reshape(B, n_ts, n_rolling * n_f)

            x = self.input_proj(x) + self.pos_emb_series

            x = self.encoder(x, sparsify=self.sparsify)

        if self.roll_y == True and self.attn_direction == "T":
            return self.output_head(x)

        else:
            return self.output_head(x[:, -1, :])


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
            roll_y=False
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
        self.pos_emb_time = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb_series = nn.Parameter(torch.randn(1, 1, d_model))
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

    def series_attention(self, x, attn_layer):
        """Cross-sectional attention: attend over the N series dimension independently for each (B,T)"""
        B, T, N, D = x.shape
        x_flat = x.view(B * T, N, D)
        out = attn_layer.self_attn(x_flat, x_flat, x_flat)
        out = out.view(B, T, N, D)
        return out

    def temporal_attention(self, x, attn_encoder):
        """Temporal attention: attend over the T time dimension independently for each series"""
        B, T, N, D = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(B * N, T, D)
        out = attn_encoder(x_flat, mask=self.mask, sparsify=self.sparsify)
        out = out.view(B, N, T, D).transpose(1, 2)
        return out

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, _ = x.shape
        x = self.input_proj(x)

        # Add learnable positional embeddings (broadcasted over B, T, N)
        x = x + self.pos_emb_time + self.pos_emb_series
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
