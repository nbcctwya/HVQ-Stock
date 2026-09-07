"""Validation callbacks whose formal state starts at a configured epoch."""

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class WarmupAwareModelCheckpoint(ModelCheckpoint):
    """Ignore all checkpoint candidates before ``start_epoch``."""

    def __init__(self, *args, start_epoch, **kwargs):
        super().__init__(*args, **kwargs)
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        self.start_epoch = int(start_epoch)

    def _formal_phase_started(self, trainer):
        return trainer.current_epoch >= self.start_epoch

    def on_validation_end(self, trainer, pl_module):
        if self._formal_phase_started(trainer):
            super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._formal_phase_started(trainer):
            super().on_train_epoch_end(trainer, pl_module)


class WarmupAwareEarlyStopping(EarlyStopping):
    """Start best-score and patience accounting only at ``start_epoch``."""

    def __init__(self, *args, start_epoch, **kwargs):
        super().__init__(*args, **kwargs)
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        self.start_epoch = int(start_epoch)

    def _formal_phase_started(self, trainer):
        return trainer.current_epoch >= self.start_epoch

    def on_validation_end(self, trainer, pl_module):
        if self._formal_phase_started(trainer):
            super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._formal_phase_started(trainer):
            super().on_train_epoch_end(trainer, pl_module)
