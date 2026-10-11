"""Функция потерь CRCE по формуле (2.27) главы 2."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss для эмбеддингов патчей.

    Каждый патч наследует метку временного окна, которому принадлежит.
    Положительными считаются все остальные патчи того же класса.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if temperature <= 0.0:
            raise ValueError(
                "temperature должна быть положительной"
            )

        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings должны иметь форму "
                "[batch, num_patches, d_model]"
            )

        batch_size, number_of_patches, embedding_dim = (
            embeddings.shape
        )

        if targets.shape != (batch_size,):
            raise ValueError(
                "targets должны иметь форму [batch]"
            )

        features = embeddings.reshape(
            batch_size * number_of_patches,
            embedding_dim,
        )
        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

        patch_targets = targets.repeat_interleave(
            number_of_patches
        )

        logits = torch.matmul(
            features,
            features.transpose(0, 1),
        ) / self.temperature

        # Вычитание максимума не меняет softmax,
        # но предотвращает переполнение exp.
        logits = logits - logits.max(
            dim=1,
            keepdim=True,
        ).values.detach()

        number_of_features = features.shape[0]
        not_self = ~torch.eye(
            number_of_features,
            dtype=torch.bool,
            device=features.device,
        )

        same_class = patch_targets.unsqueeze(0).eq(
            patch_targets.unsqueeze(1)
        )
        positive_mask = same_class & not_self

        exp_logits = torch.exp(logits) * not_self
        log_denominator = torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(self.eps)
        )
        log_probability = logits - log_denominator

        positive_count = positive_mask.sum(dim=1)
        valid_anchors = positive_count > 0

        if not valid_anchors.any():
            # Сохраняем связь с графом вычислений.
            return embeddings.sum() * 0.0

        mean_positive_log_probability = (
            (log_probability * positive_mask).sum(dim=1)
            / positive_count.clamp_min(1)
        )

        return -mean_positive_log_probability[
            valid_anchors
        ].mean()


class CRCELoss(nn.Module):
    """CE + lambda1 * L2 + lambda2 * SupCon."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        lambda_l2: float = 1e-4,
        lambda_supcon: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()

        if lambda_l2 < 0.0 or lambda_supcon < 0.0:
            raise ValueError(
                "Коэффициенты потерь не могут быть отрицательными"
            )

        self.lambda_l2 = lambda_l2
        self.lambda_supcon = lambda_supcon
        self.register_buffer(
            "class_weights",
            None
            if class_weights is None
            else class_weights.detach().clone(),
        )
        self.supcon = SupervisedContrastiveLoss(
            temperature=temperature
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
        )

        l2_loss = torch.zeros(
            (),
            device=logits.device,
            dtype=logits.dtype,
        )
        for parameter in model.parameters():
            if parameter.requires_grad:
                l2_loss = l2_loss + parameter.pow(2).sum()

        supcon_loss = self.supcon(
            embeddings,
            targets,
        )

        total_loss = (
            ce_loss
            + self.lambda_l2 * l2_loss
            + self.lambda_supcon * supcon_loss
        )

        components = {
            "ce": ce_loss.detach(),
            "l2": l2_loss.detach(),
            "supcon": supcon_loss.detach(),
            "weighted_l2": (
                self.lambda_l2 * l2_loss
            ).detach(),
            "weighted_supcon": (
                self.lambda_supcon * supcon_loss
            ).detach(),
        }

        return total_loss, components


if __name__ == "__main__":
    torch.manual_seed(777)

    test_model = nn.Linear(8, 3)
    test_input = torch.randn(4, 8)
    test_logits = test_model(test_input)
    test_embeddings = torch.randn(
        4,
        7,
        8,
        requires_grad=True,
    )
    test_targets = torch.tensor([0, 0, 1, 1])

    criterion = CRCELoss(
        lambda_l2=1e-4,
        lambda_supcon=0.1,
        temperature=0.07,
    )

    loss, parts = criterion(
        test_logits,
        test_targets,
        test_embeddings,
        test_model,
    )
    loss.backward()

    print(f"Total: {loss.item():.6f}")
    print(f"CE: {parts['ce'].item():.6f}")
    print(f"L2: {parts['weighted_l2'].item():.6f}")
    print(
        "SupCon: "
        f"{parts['weighted_supcon'].item():.6f}"
    )
    print(
        "Finite: "
        f"{torch.isfinite(loss).item()}"
    )
    print(
        "Embedding gradient: "
        f"{test_embeddings.grad is not None}"
    )
