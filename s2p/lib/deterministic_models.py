import torch
import torch.nn as nn

class DeterministicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[l for _ in range(hidden_layers) for l in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
            nn.Linear(hidden_dim, output_dim),
        )
        self.to(self.device)
        
    def forward(self, x):
        return self.net(x)
    
class DeterministicCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_layers, output_dim, device):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.device = device

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            *[l for _ in range(hidden_layers) for l in (
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=-3),
            nn.Linear(hidden_channels, output_dim),
        )
        self.to(self.device)

    def forward(self, x):
        return self.net(x)


class _CAPBlock(nn.Module):
    """One cross-attention + feedforward sublayer of a DeterministicCAP stack."""

    def __init__(self, token_dim, mha_embedding_dim, mha_n_heads,
                 ff_hidden_layers, ff_hidden_dim, device):
        super().__init__()
        self.kv_proj = nn.Linear(token_dim, mha_embedding_dim)
        self.cross_attn = nn.MultiheadAttention(mha_embedding_dim, mha_n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(mha_embedding_dim)
        self.ff = DeterministicMLP(
            input_dim=mha_embedding_dim,
            hidden_dim=ff_hidden_dim,
            hidden_layers=ff_hidden_layers,
            output_dim=mha_embedding_dim,
            device=device,
        )
        self.ff_norm = nn.LayerNorm(mha_embedding_dim)
        self.to(device)

    def forward(self, q, x):
        kv = self.kv_proj(x)
        attn_out, _ = self.cross_attn(q, kv, kv)
        q = self.attn_norm(q + attn_out)   # residual MHA
        q = self.ff_norm(q + self.ff(q))   # residual FF
        return q


class DeterministicCAP(nn.Module):
    """
    Cross-Attention Pooling: n_queries learnable tokens attend over input tokens, each
    sublayer (the cross-attention itself, then a feedforward block) wrapped in a residual
    connection and LayerNorm, as in a standard Transformer block — this is what gives the
    pooled queries a nonlinearity beyond the linear/softmax mixing of attention alone.

    n_blocks stacks n_blocks of these blocks: every block cross-attends over the same
    input tokens, but block i's output queries (rather than a fresh set of learnable
    queries) become the query input to block i+1, so the pooled queries are refined
    progressively across the stack instead of each block pooling the input independently.
    """

    def __init__(self,
                 token_dim,
                 mha_embedding_dim, mha_n_heads, output_dim, mha_n_queries=1,
                 ff_hidden_layers=2, ff_hidden_dim=512,
                 head_hidden_layers=2, head_hidden_dim=512,
                 n_blocks=1,
                 device=None):
        super().__init__()
        self.token_dim = token_dim
        self.queries = nn.Parameter(torch.randn(1, mha_n_queries, mha_embedding_dim))
        self.blocks = nn.ModuleList([
            _CAPBlock(
                token_dim=token_dim,
                mha_embedding_dim=mha_embedding_dim,
                mha_n_heads=mha_n_heads,
                ff_hidden_layers=ff_hidden_layers,
                ff_hidden_dim=ff_hidden_dim,
                device=device,
            )
            for _ in range(n_blocks)
        ])
        # self.out = nn.Linear(mha_n_queries * mha_embedding_dim, output_dim)
        self.out = DeterministicMLP(
                input_dim=mha_n_queries * mha_embedding_dim,
                hidden_dim=head_hidden_dim,
                hidden_layers=head_hidden_layers,
                output_dim=output_dim,
                device=device,
            )
        self.output_dim = output_dim
        self.to(device)

    def forward(self, x):
        # x: (B, N, token_dim)  — explicit token sequence
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, x)
        return self.out(q.flatten(1))