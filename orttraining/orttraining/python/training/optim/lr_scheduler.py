import torch
import math

from . import config


class LRScheduler(object):
    r"""Base class for implementing custom learning rate schedulers

    Once the scheduler is configured, no user code is needed to update learning rate.

    NOTE: This class should never be instantiated, but used as an abstract class.

    Args:
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.Adam`, :py:class:`.optim.Lamb` or :py:class:`.optim.SGD`.
    """

    def __init__(self, optimizer_config):
        assert isinstance(optimizer_config, config._OptimizerConfig),\
            "optimizer_config must be :py:class:`.optim._OptimizerConfig"
        self.optimizer_config = optimizer_config

    def get_lr(self, train_step_info):
        r"""Returns a list of learning rate

        Args:
            train_step_info (:py:class:`.TrainStepInfo`): runtime info for current training step

        Returns:
            ordered :py:obj:`list` of learning rates.
                The first entry is the default learning rate and
                    the remaining, refer to each :py:class:`optim._OptimizerConfig`s parameter group.
            NOTE: Currently, only default learning rate is supported,
                which implies returning a list with a single learning rate.
        """
        raise NotImplementedError


class ConstantWarmupLRScheduler(LRScheduler):
    r"""Constant warmup strategy for learning rate update

    Learning rate update strategy:
        lr = base_lr * (step / total_steps) / warmup, when step / total_steps < warmup
        lr = base_lr, when step / total_steps >= warmup

    Args:
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.Adam`, :py:class:`.optim.Lamb` or :py:class:`.optim.SGD`.
        total_steps (int): total training steps for learning.
        warmup (float, default is 0.002): portion of total steps for warmup.

    Example:
        .. code-block:: python

            # Initialize optimizer config
            optimizer_config = optim.SGD(lr=0.001)

            # Initialize lr scheduler
            lr_scheduler = ConstantWarmupLRScheduler(optimizer_config, total_steps=512, warmup=0.002)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, optimizer_config, total_steps, warmup=0.002):
        super().__init__(optimizer_config)
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(warmup, float) and warmup >= 0,\
            "warmup must be a positive float"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.warmup = warmup

    def _warmup_constant(self, train_step_info):
        x = train_step_info.step / self.total_steps
        if x < self.warmup:
            return x/self.warmup
        return 1.0

    def get_lr(self, train_step_info):
        return [self.optimizer_config.lr * self._warmup_constant(train_step_info)]


class CosineWarmupLRScheduler(LRScheduler):
    r"""Cosine warmup strategy for learning rate update

    Learning rate update strategy:
        lr = base_lr * (step / total_steps) / warmup, when step / total_steps < warmup
        lr = base_lr * 0.5 * (1.0 + cosine(pi * (step / total_steps))), when step / total_steps >= warmup

    Args:
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.Adam`, :py:class:`.optim.Lamb` or :py:class:`.optim.SGD`.
        total_steps (int): total training steps for learning.
        warmup (float, default is 0.002): portion of total steps for warmup.

    Example:
        .. code-block:: python

            # Initialize optimizer config
            optimizer_config = optim.SGD(lr=0.001)

            # Initialize lr scheduler
            lr_scheduler = CosineWarmupLRScheduler(optimizer_config, total_steps=512, warmup=0.002)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, optimizer_config, total_steps, warmup=0.002):
        super().__init__(optimizer_config)
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(warmup, float) and warmup >= 0,\
            "warmup must be a positive float"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.warmup = warmup

    def _warmup_cosine(self, train_step_info):
        x = train_step_info.step / self.total_steps
        if x < self.warmup:
            return x/self.warmup
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    def get_lr(self, train_step_info):
        return [self.optimizer_config.lr * self._warmup_cosine(train_step_info)]


class LinearWarmupLRScheduler(LRScheduler):
    r"""Linear warmup strategy for learning rate update

    Learning rate update strategy:
        lr = base_lr * (step / total_steps) / warmup, when step / total_steps < warmup
        lr = base_lr * max(((step / total_steps) - 1.) / (warmup - 1.), 0.), when step / total_steps >= warmup

    Args:
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.Adam`, :py:class:`.optim.Lamb` or :py:class:`.optim.SGD`.
        total_steps (int): total training steps for learning.
        warmup (float, default is 0.002): portion of total steps for warmup.

    Example:
        .. code-block:: python

            # Initialize optimizer config
            optimizer_config = optim.SGD(lr=0.001)

            # Initialize lr scheduler
            lr_scheduler = LinearWarmupLRScheduler(optimizer_config, total_steps=512, warmup=0.002)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, optimizer_config, total_steps, warmup=0.002):
        super().__init__(optimizer_config)
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(warmup, float) and warmup >= 0,\
            "warmup must be a positive float"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.warmup = warmup

    def _warmup_linear(self, train_step_info):
        x = train_step_info.step / self.total_steps
        if x < self.warmup:
            return x / self.warmup
        return max((x - 1.) / (self.warmup - 1.), 0.)

    def get_lr(self, train_step_info):
        return [self.optimizer_config.lr * self._warmup_linear(train_step_info)]


class PolyWarmupLRScheduler(LRScheduler):
    r"""Polynomial warmup strategy for learning rate update

    Learning rate update strategy:
        lr = base_lr * (step / total_steps) / warmup, when step / total_steps < warmup
        lr = base_lr * (1 − step / total_steps ) ^ degree, when step / total_steps >= warmup

    Args:
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.Adam`, :py:class:`.optim.Lamb` or :py:class:`.optim.SGD`.
        total_steps (int): total training steps for learning.
        warmup (float, default is 0.002): portion of total steps for warmup.
        degree (float, default is 0.5): polynomial power

    Example:
        .. code-block:: python

            # Initialize optimizer config
            optimizer_config = optim.SGD(lr=0.001)

            # Initialize lr scheduler
            lr_scheduler = PolyWarmupLRScheduler(optimizer_config, total_steps=512, warmup=0.002, degree=0.5)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, optimizer_config, total_steps, warmup=0.002, degree=0.5):
        super().__init__(optimizer_config)
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(warmup, float) and warmup >= 0,\
            "warmup must be a positive float"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"
        assert isinstance(degree, float) and warmup >= 0,\
            "degree must be a positive float"

        self.total_steps = total_steps
        self.warmup = warmup
        self.degree = degree

    def _warmup_poly(self, train_step_info):
        x = train_step_info.step / self.total_steps
        if x < self.warmup:
            return x/self.warmup
        return (1.0 - x)**self.degree

    def get_lr(self, train_step_info):
        return [self.optimizer_config.lr * self._warmup_poly(train_step_info)]
