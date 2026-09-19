"""
Scheduled Sampling for World Dynamics Transformer Training.

Implements curriculum scheduling between teacher-forced training and autoregressive
generation (Bengio et al., 2015) to mitigate exposure bias and prevent error explosion
during multi-step rollouts.

Schedules:
- Linear: p_sample = min(p_max, p_start + (p_max - p_start) * (epoch / warmup_epochs))
- Sigmoidal: p_sample = p_max / (1 + exp((half_epoch - epoch) / k))
"""

import math
import random
import torch


class ScheduledSampler:
    """
    Computes sampling probability p_sample for scheduled sampling at each epoch
    and provides helper to mix ground-truth and model-predicted tokens.
    """

    def __init__(
        self,
        p_start: float = 0.0,
        p_max: float = 0.5,
        anneal_epochs: int = 20,
        schedule_type: str = "linear",
    ):
        self.p_start = p_start
        self.p_max = p_max
        self.anneal_epochs = anneal_epochs
        self.schedule_type = schedule_type
        self.current_p = p_start

    def step_epoch(self, epoch: int) -> float:
        """Update and return current p_sample for the given epoch (1-indexed)."""
        if self.schedule_type == "linear":
            progress = min(1.0, max(0.0, float(epoch - 1) / max(1, self.anneal_epochs)))
            self.current_p = self.p_start + (self.p_max - self.p_start) * progress
        elif self.schedule_type == "exponential":
            decay = self.p_max ** (1.0 / max(1, self.anneal_epochs))
            self.current_p = min(self.p_max, self.p_start + (decay ** (self.anneal_epochs - epoch)))
        else:
            self.current_p = self.p_max if epoch > self.anneal_epochs else self.p_start

        return self.current_p

    def should_sample_prediction(self) -> bool:
        """Returns True with probability current_p."""
        return random.random() < self.current_p

    def mix_tokens(
        self,
        gt_token: torch.Tensor,
        pred_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples between ground truth and model prediction.
        Model prediction is detached to prevent invalid gradient feedback loops.
        """
        if self.should_sample_prediction():
            return pred_token.detach()
        return gt_token
