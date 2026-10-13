"""Neural classifiers comparing pooling of frozen Point-JEPA patch features.

The baseline uses the paper's max-plus-mean pooling with a neural MLP head,
rather than its linear SVM. Attention pooling operates after the Transformer.
"""

import torch
from torch import nn


def _float_features(x, feature_dim):
    if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != feature_dim:
        raise ValueError(
            f"Expected [B, patches, {feature_dim}] with at least one patch, "
            f"got {tuple(x.shape)}"
        )
    # Cached features are float16; keep pooling and classifier arithmetic float32.
    return x.float()


class MaxMeanPooling(nn.Module):
    def __init__(self, feature_dim=384):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        x = _float_features(x, self.feature_dim)
        # Original pooling: concatenate patchwise maximum and arithmetic mean.
        return torch.cat([x.max(dim=1).values, x.mean(dim=1)], dim=-1)


class MaxAttentionPooling(nn.Module):
    def __init__(self, feature_dim=384):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x, return_attention=False):
        x = _float_features(x, self.feature_dim)
        # Research modification: replace the mean with learnable patch weights.
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        attention_features = (x * weights.unsqueeze(-1)).sum(dim=1)
        global_features = torch.cat(
            [x.max(dim=1).values, attention_features], dim=-1
        )
        if return_attention:
            return global_features, weights
        return global_features


class MaxMinStdPooling(nn.Module):
    """Concatenate patchwise maximum, minimum, and population deviation."""

    def __init__(self, feature_dim=384):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        x = _float_features(x, self.feature_dim)
        return torch.cat([x.max(dim=1).values, x.min(dim=1).values,
                          x.std(dim=1, unbiased=False)], dim=-1)


class ClassificationHead(nn.Module):
    """MLP head; default input preserves existing checkpoint tensor shapes."""

    def __init__(self, feature_dim=384, hidden_dim=256, num_classes=40, dropout=0.2,
                 input_dim=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * feature_dim if input_dim is None else input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, global_features):
        return self.layers(global_features.float())


class MaxMeanClassifier(nn.Module):
    """Neural baseline; forward returns raw [B, num_classes] logits."""

    def __init__(self, feature_dim=384, hidden_dim=256, num_classes=40, dropout=0.2):
        super().__init__()
        self.pooling = MaxMeanPooling(feature_dim)
        self.head = ClassificationHead(feature_dim, hidden_dim, num_classes, dropout)

    def forward(self, x):
        return self.head(self.pooling(x))


class MaxAttentionClassifier(nn.Module):
    """Return logits, or (logits, [B, patches] weights) when requested."""

    def __init__(self, feature_dim=384, hidden_dim=256, num_classes=40, dropout=0.2):
        super().__init__()
        self.pooling = MaxAttentionPooling(feature_dim)
        self.head = ClassificationHead(feature_dim, hidden_dim, num_classes, dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            global_features, weights = self.pooling(x, return_attention=True)
            return self.head(global_features), weights
        return self.head(self.pooling(x))


class MaxMinStdClassifier(nn.Module):
    """Frozen-feature classifier returning raw [B, num_classes] logits."""

    def __init__(self, feature_dim=384, hidden_dim=256, num_classes=40, dropout=0.2):
        super().__init__()
        self.pooling = MaxMinStdPooling(feature_dim)
        self.head = ClassificationHead(feature_dim, hidden_dim, num_classes, dropout,
                                       input_dim=3 * feature_dim)

    def forward(self, x):
        return self.head(self.pooling(x))


MODEL_CLASSES = {"baseline": MaxMeanClassifier, "attention": MaxAttentionClassifier,
                 "max_min_std": MaxMinStdClassifier}


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def smoke_test():
    """Check both models using synthetic cached-style inputs on the CPU."""
    torch.manual_seed(42)
    x = torch.randn(4, 64, 384, dtype=torch.float16)
    baseline = MaxMeanClassifier()
    proposed = MaxAttentionClassifier()
    baseline.eval()
    proposed.eval()

    with torch.no_grad():
        baseline_logits = baseline(x)
        attention_logits, weights = proposed(x, return_attention=True)
        assert baseline.pooling(x).shape == (4, 768)
        assert proposed.pooling(x).shape == (4, 768)
        for logits in (baseline_logits, attention_logits):
            assert logits.shape == (4, 40)
            assert logits.dtype == torch.float32
            assert torch.isfinite(logits).all()
        assert weights.shape == (4, 64)
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6)

    proposed.train()
    proposed.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(proposed(x), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    for parameter in proposed.pooling.attention.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    gradient = proposed.pooling.attention.weight.grad
    assert gradient.abs().sum().item() > 0
    # The bias-free score weights must learn patch preferences through gradients.

    print(f"Input: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    print("Pooled features: (4, 768) for both models")
    print(f"Baseline logits: {tuple(baseline_logits.shape)}; finite=True")
    print(f"Attention logits: {tuple(attention_logits.shape)}; finite=True")
    print(f"Baseline trainable parameters: {count_trainable_parameters(baseline):,}")
    print(f"Attention trainable parameters: {count_trainable_parameters(proposed):,}")
    print(f"Attention weights: {tuple(weights.shape)}; nonnegative=True")
    print(f"Attention sums per sample: {weights.sum(dim=1).tolist()}; normalized=True")
    print(f"Attention score-weight gradient L1 norm: {gradient.abs().sum().item():.6f}")
    print("Attention parameters receive gradients; score-weight gradients are nonzero.")
    new_model = MaxMinStdClassifier()
    pooled = new_model.pooling(x)
    assert pooled.shape == (4, 1152)
    assert pooled.dtype == torch.float32 and torch.isfinite(pooled).all()
    # Known patch statistics distinguish dim=1 from featurewise deviation and
    # population deviation from the sample (unbiased) estimator.
    known = torch.arange(64).float().view(1, 64, 1).expand(2, 64, 384)
    known_pool = new_model.pooling(known)
    assert torch.equal(known_pool[:, :384], torch.full((2, 384), 63.0))
    assert torch.equal(known_pool[:, 384:768], torch.zeros(2, 384))
    assert torch.allclose(known_pool[:, 768:], torch.full((2, 384), (4095 / 12) ** 0.5))
    assert torch.equal(new_model.pooling(torch.ones(2, 1, 384))[:, 768:],
                       torch.zeros(2, 384))
    logits = new_model(x)
    assert logits.shape == (4, 40) and torch.isfinite(logits).all()
    assert count_trainable_parameters(new_model) == 305448
    for model in (baseline, new_model):
        model.train()
        model.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(x), torch.arange(4)).backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert sum(p.grad.abs().sum().item() for p in model.parameters()) > 0
    # Pooling itself also propagates finite gradients to input features.
    grad_input = x.float().requires_grad_()
    new_model.pooling(grad_input).sum().backward()
    assert torch.isfinite(grad_input.grad).all()
    print("Max/Min/Std: (4, 64, 384) -> (4, 1152) -> (4, 40)")
    print("Max/Min/Std trainable parameters: 305,448; finite outputs/gradients=True")
    print("Patchwise population std and singleton-patch zero std verified.")
    print("Smoke test passed for all three models.")


if __name__ == "__main__":
    smoke_test()
