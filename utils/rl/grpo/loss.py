import torch
import torch.nn as nn
from typing import Optional
from utils.rl.grpo.replay import ExperienceHist
from utils.rl.grpo.functions import get_batch_log_probs


def kl_div(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """
    log_probs_ref = log_probs_ref.to(device=log_probs.device)
    mask = mask.to(device=log_probs.device)
    log_ratio = log_probs_ref.float() - log_probs.float()
    if mask is not None:
        log_ratio = log_ratio * mask

    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        rwkv,
        experience_hist: ExperienceHist,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        log_probs = get_batch_log_probs(
            rwkv=rwkv,
            t_batch_tokens=experience_hist.history_tokens,
            begin_with_states=experience_hist.begin_with_states,
        )

        old_log_probs = experience_hist.action_log_probs
        log_probs_ref = experience_hist.log_probs_ref
        action_mask = experience_hist.action_mask
        advantages = experience_hist.advantages

        kl = kl_div(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()
