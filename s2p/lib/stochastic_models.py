import torch
import torch.nn as nn

from s2p.lib.deterministic_models import DeterministicMLP, _CAPBlock

# Default clamp on the predicted log-std, keeps the output distribution's scale
# in a numerically stable range (standard practice in continuous-control RL).
LOG_STD_MIN = -7.0
LOG_STD_MAX = 2.0


def _make_heads(input_dim, output_dim, head_hidden_dim, head_hidden_layers, device):
    """Two small nonlinear heads (mean, log_std) mapping input_dim -> output_dim."""
    mean_head = DeterministicMLP(
        input_dim=input_dim,
        hidden_dim=head_hidden_dim,
        hidden_layers=head_hidden_layers,
        output_dim=output_dim,
        device=device,
    )
    log_std_head = DeterministicMLP(
        input_dim=input_dim,
        hidden_dim=head_hidden_dim,
        hidden_layers=head_hidden_layers,
        output_dim=output_dim,
        device=device,
    )
    return mean_head, log_std_head


class StochasticMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim,
                 head_hidden_dim, head_hidden_layers, device,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_hidden_layers = head_hidden_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[l for _ in range(hidden_layers) for l in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())],
        )
        self.mean_head, self.log_std_head = _make_heads(
            hidden_dim, output_dim, head_hidden_dim, head_hidden_layers, device)
        self.to(self.device)

    def forward(self, x):
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std.exp()


class StochasticCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_layers, output_dim,
                 head_hidden_dim, head_hidden_layers, device,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_hidden_layers = head_hidden_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            *[l for _ in range(hidden_layers) for l in (
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=-3),
        )
        self.mean_head, self.log_std_head = _make_heads(
            hidden_channels, output_dim, head_hidden_dim, head_hidden_layers, device)
        self.to(self.device)

    def forward(self, x):
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std.exp()


VAR_MIN = 1e-12  # floor on pooled variances, keeps the sqrt (and its gradient) finite

POOLINGS = ("mean", "attention")
AGGREGATIONS = ("mixture", "parameters", "independent", "product")


class _TokenWeights(nn.Module):
    """
    Per-token pooling weights w, normalised over the sequence (sum_i w_i = 1).

    'mean'      — uniform 1/N, a plain average over tokens.
    'attention' — a learned scalar score per token, softmaxed over the sequence, so the
                  model can decide which tokens its latent should be read from.
    """

    def __init__(self, pooling, feature_dim, device=None):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"Unknown pooling '{pooling}', expected one of {POOLINGS}.")
        self.pooling = pooling
        self.score = nn.Linear(feature_dim, 1) if pooling == "attention" else None
        self.to(device)

    def forward(self, h):
        # h: (B, N, feature_dim) -> (B, N, 1)
        if self.pooling == "attention":
            return torch.softmax(self.score(h), dim=1)
        return h.new_full((h.shape[0], h.shape[1], 1), 1.0 / h.shape[1])


def _aggregate_gaussians(aggregation, mean, var, weights):
    """
    Combine the per-token Gaussians N(mean_i, var_i), weighted by `weights` (which sum to
    1 over the token dimension), into a single diagonal Gaussian.

    All inputs are (B, N, output_dim) except `weights`, which broadcasts as (B, N, 1);
    returns (mean, var) of shape (B, output_dim).

    'mixture'     — the equally/attention-weighted mixture of the token distributions,
                    moment-matched back to a Gaussian. The pooled variance is the average
                    token variance *plus* the spread of the token means, so tokens
                    disagreeing about the latent shows up as uncertainty.
    'parameters'  — average the distribution parameters directly (mean of means, mean of
                    standard deviations). Cheapest reading of "average the distributions";
                    ignores cross-token disagreement.
    'independent' — the distribution of the weighted average of one independent sample per
                    token: var = sum_i w_i^2 var_i. Note this shrinks the scale by ~1/N,
                    so with many tokens the latent becomes near-deterministic.
    'product'     — product of experts, each token tempered by its weight, so the pooled
                    precision is the weighted *average* of the token precisions (not their
                    sum) and the scale stays independent of N. Confident tokens dominate:
                    the pooled variance is the weighted harmonic mean of the token
                    variances, so it is pulled towards the smallest of them.
    """
    if aggregation == "mixture":
        pooled_mean = (weights * mean).sum(dim=1)
        second_moment = (weights * (var + mean.square())).sum(dim=1)
        return pooled_mean, second_moment - pooled_mean.square()
    if aggregation == "parameters":
        pooled_mean = (weights * mean).sum(dim=1)
        pooled_std = (weights * var.sqrt()).sum(dim=1)
        return pooled_mean, pooled_std.square()
    if aggregation == "independent":
        return (weights * mean).sum(dim=1), (weights.square() * var).sum(dim=1)
    if aggregation == "product":
        precision = (weights / var).sum(dim=1)
        pooled_var = 1.0 / precision
        return pooled_var * (weights * mean / var).sum(dim=1), pooled_var
    raise ValueError(f"Unknown aggregation '{aggregation}', expected one of {AGGREGATIONS}.")


class StochasticPooling(nn.Module):
    """
    Permutation-invariant stochastic token encoder: every token is passed independently
    through a shared MLP whose two heads map *that token* to a diagonal Gaussian over the
    output space, and the resulting per-token distributions are pooled over the sequence
    into a single diagonal Gaussian.

    This is the cheap counterpart to StochasticCAP — no attention between queries and
    tokens, just a DeepSets-style encode-then-pool — and unlike StochasticCAP the
    stochasticity is produced per token rather than after pooling. Both the token
    weighting (`pooling`) and the rule combining the token distributions (`aggregation`)
    are configurable; see `_TokenWeights` and `_aggregate_gaussians`.
    """

    def __init__(self,
                 token_dim,
                 output_dim,
                 token_hidden_dim=512, token_hidden_layers=2, token_output_dim=512,
                 pooling="mean", aggregation="mixture",
                 head_hidden_dim=256, head_hidden_layers=1,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX,
                 device=None):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation '{aggregation}', expected one of {AGGREGATIONS}.")
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.token_output_dim = token_output_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_hidden_layers = head_hidden_layers
        self.pooling = pooling
        self.aggregation = aggregation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.token_mlp = DeterministicMLP(
            input_dim=token_dim,
            hidden_dim=token_hidden_dim,
            hidden_layers=token_hidden_layers,
            output_dim=token_output_dim,
            device=device,
        )
        # Applied per token, so each token gets its own distribution over the output space.
        self.mean_head, self.log_std_head = _make_heads(
            token_output_dim, output_dim, head_hidden_dim, head_hidden_layers, device)
        self.weights = _TokenWeights(pooling, token_output_dim, device=device)
        self.to(device)

    def token_distribution_parameters(self, x):
        """Per-token (mean, std, pooling weight); shapes (B, N, output_dim) and (B, N, 1)."""
        h = self.token_mlp(x)              # (B, N, token_output_dim), MLP applied per token
        mean = self.mean_head(h)           # (B, N, output_dim)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std.exp(), self.weights(h)

    def forward(self, x):
        # x: (B, N, token_dim)  — explicit token sequence
        mean, std, weights = self.token_distribution_parameters(x)
        pooled_mean, pooled_var = _aggregate_gaussians(
            self.aggregation, mean, std.square(), weights)
        return pooled_mean, pooled_var.clamp_min(VAR_MIN).sqrt()


class StochasticCAP(nn.Module):
    """
    Stochastic counterpart of DeterministicCAP: identical cross-attention pooling
    stack, but the single linear output projection is replaced by two small nonlinear
    heads that parameterise a diagonal Gaussian over the output (mean and log-std).
    """

    def __init__(self,
                 token_dim,
                 mha_embedding_dim, mha_n_heads, output_dim, mha_n_queries=1,
                 ff_hidden_layers=1, ff_hidden_dim=512,
                 n_blocks=1,
                 head_hidden_dim=256, head_hidden_layers=1,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX,
                 device=None):
        super().__init__()
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_hidden_layers = head_hidden_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
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
        self.mean_head, self.log_std_head = _make_heads(
            mha_n_queries * mha_embedding_dim, output_dim,
            head_hidden_dim, head_hidden_layers, device)
        self.to(device)

    def forward(self, x):
        # x: (B, N, token_dim)  — explicit token sequence
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, x)
        h = q.flatten(1)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std
