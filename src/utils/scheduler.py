from torch.optim.lr_scheduler import _LRScheduler


class WarmUpScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        feature_size: int,
        factor: float = 1.0,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.feature_size = feature_size
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self._compute_lr()
        return [lr] * len(self.base_lrs)

    def _compute_lr(self):
        if self.last_epoch == 0:
            return 0.0

        lr = (self.feature_size ** (-0.5)) * min(
            self.last_epoch ** (-0.5), self.last_epoch * self.warmup_steps ** (-1.5)
        )

        return lr * self.factor
