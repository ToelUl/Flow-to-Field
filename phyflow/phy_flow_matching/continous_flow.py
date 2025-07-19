# -*- coding: utf-8 -*-
"""
This module implements core components for training and utilizing
Continuous Flow Matching (CFM) models, including data handling utilities,
adaptive gradient clipping, and an executor class for managing the
training, likelihood computation, and sampling processes.
"""

# --- Existing Imports ---
import random
import time
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Iterator, Any, Type, Union, Callable

import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.distributions import Normal, Distribution, Independent

from .tools import logit_normal_sampler, timestep_scheduler
from .models import CFMWrapper

from flow_matching.path.scheduler import (CondOTScheduler, ConvexScheduler,
                                          ScheduleTransformedModel)
from flow_matching.solver import ODESolver
from flow_matching.path import AffineProbPath
from flow_matching.utils import ModelWrapper

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

torch.set_float32_matmul_precision('high')

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def multi_dataloader_random_cycle(
    dataloaders: Tuple[DataLoader, ...],
    num_iterations: int
) -> Iterator[Any]:
    """Randomly cycles through multiple DataLoaders for a fixed number of iterations.

    This iterator continuously yields batches by randomly selecting one of the
    provided DataLoaders. If a selected DataLoader is exhausted, it is
    automatically reset and re-used. The iteration stops after `num_iterations`
    batches have been yielded.

    Args:
        dataloaders: A tuple of PyTorch DataLoader objects.
        num_iterations: The total number of batches to yield.

    Yields:
        A batch of data randomly selected from one of the DataLoaders.

    Raises:
        StopIteration: When `num_iterations` have been yielded.
    """
    if not dataloaders: # Handle empty dataloaders tuple
        return

    iterators = [iter(dl) for dl in dataloaders]

    for _ in range(num_iterations):
        # This check is more of a safeguard; with the current logic of
        # resetting iterators, this list should not become empty unless
        # dataloaders was initially empty or all dataloaders become permanently empty.
        if not iterators:
            break

        idx = random.randrange(len(iterators))
        try:
            batch = next(iterators[idx])
        except StopIteration:
            # Reset the exhausted iterator to allow continuous cycling
            logger.debug(f"DataLoader at index {idx} exhausted. Resetting.")
            iterators[idx] = iter(dataloaders[idx])
            try:
                batch = next(iterators[idx])
            except StopIteration:
                # This can happen if a DataLoader is inherently empty even after reset
                logger.warning(
                    f"DataLoader at index {idx} seems empty even after reset. "
                    "Skipping this selection."
                )
                continue  # Skip this iteration and try another DataLoader
        yield batch


class AutoClip:
    """Adaptive gradient clipping based on percentile of historical gradient norms.

    This class implements a strategy for adaptive gradient clipping. Instead of
    using a fixed clipping threshold, it dynamically calculates the threshold
    based on a specified percentile of the L2 norms of gradients observed
    during training. This can help stabilize training by preventing exploding
    gradients while adapting to the changing scale of gradients.

    Attributes:
        model: The PyTorch model whose gradients will be clipped.
        percentile: The percentile (0-100) used to determine the clipping
            threshold from the history of gradient norms.
        window_size: The maximum number of recent gradient norms to store in
            the history. If None, all historical norms are used.
        grad_history: A list storing the L2 norms of gradients from recent
            optimizer steps.
    """

    def __init__(self,
                 model: nn.Module,
                 percentile: float = 10.0,
                 window_size: Optional[int] = None) -> None:
        """Initializes the AutoClip instance.

        Args:
            model: The model whose gradients will be clipped.
            percentile: Percentile (0-100) to compute the clipping threshold.
                For example, a percentile of 10.0 means the gradients will be
                clipped at the 10th percentile of the observed gradient norms.
            window_size: If provided, only the most recent `window_size`
                gradient norms are used for percentile calculation. Otherwise,
                all historical norms are used.

        Raises:
            ValueError: If `percentile` is not between 0 and 100, or if
                `window_size` is not None and not a positive integer.
        """
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(
                f"Percentile must be between 0 and 100, got {percentile}"
            )
        if window_size is not None and window_size <= 0:
            raise ValueError(
                "window_size must be a positive integer if provided, "
                f"got {window_size}"
            )

        self.model = model
        self.percentile = percentile
        self.window_size = window_size
        self.grad_history: List[float] = []

    def _compute_grad_norm(self) -> float:
        """Computes the total L2 norm of all gradients in the model.

        Returns:
            The total L2 norm of gradients. Returns 0.0 if no gradients
            are present.
        """
        total_norm_sq = 0.0
        params_with_grad = [
            p for p in self.model.parameters() if p.grad is not None
        ]

        if not params_with_grad:
            return 0.0  # No gradients to norm

        for p in params_with_grad:
            grad_norm = p.grad.detach().norm(2).item()
            total_norm_sq += grad_norm ** 2
        return total_norm_sq ** 0.5

    def __call__(self) -> Optional[float]:
        """Computes gradient norm, updates history, and applies clipping.

        This method performs the following steps:
        1. Computes the L2 norm of the current gradients of the model.
        2. Adds this norm to the history (if positive and history is not full,
           or if windowing is active).
        3. If windowing is enabled, prunes the history to `window_size`.
        4. Calculates the clipping threshold using the specified percentile of
           the gradient norms in history.
        5. If the threshold is positive, clips the model's gradients in-place
           using `torch.nn.utils.clip_grad_norm_`.

        Returns:
            The clipping threshold value used. Returns `None` if no clipping
            was performed (e.g., empty gradient history or non-positive
            clipping threshold).
        """
        current_norm = self._compute_grad_norm()

        # Add to history only if norm is significantly positive to avoid
        # skewing percentile calculations with zeros or tiny values.
        if current_norm > 1e-6:
            self.grad_history.append(current_norm)

        # Apply windowing to the gradient history
        if self.window_size is not None and \
           len(self.grad_history) > self.window_size:
            self.grad_history = self.grad_history[-self.window_size:]

        if not self.grad_history:
            logger.debug("AutoClip: Gradient history is empty. Skipping clipping.")
            return None

        # Compute clipping threshold from percentile
        clip_val = float(np.percentile(self.grad_history, self.percentile))

        # Apply clipping if threshold is positive
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
            # logger.debug(
            #     f"AutoClip: Applied clipping with threshold {clip_val:.4f} "
            #     f"(based on {len(self.grad_history)} norms)"
            # )
            return clip_val
        else:
            logger.debug(
                f"AutoClip: Calculated clip value ({clip_val:.4f}) is not "
                "positive. Skipping clipping."
            )
            return None


class CFMExecutor:
    """Manages training, evaluation, and sampling for Continuous Flow Matching.

    This executor class encapsulates the common workflows for working with
    Continuous Flow Matching (CFM) models. It handles model training loops,
    log-likelihood computation for evaluation, and sample generation by solving
    the learned ordinary differential equation (ODE). It utilizes standard
    `torch.distributions` for the base distribution $p_0$.

    Default Constants:
        DEFAULT_SAVE_EVERY_EPOCHS: Frequency for saving checkpoints (epochs).
        DEFAULT_NUM_STEPS: Default number of steps for fixed-step ODE solvers.
        DEFAULT_TRAIN_WITH_MLE_NUM_STEPS: Default steps for MLE training ODE solve.
        DEFAULT_METHOD: Default ODE solver method (e.g., "heun3").
        DEFAULT_TRAIN_WITH_MLE_METHOD: Default ODE solver for MLE training.
        DEFAULT_ATOL: Default absolute tolerance for adaptive ODE solvers.
        DEFAULT_RTOL: Default relative tolerance for adaptive ODE solvers.
        DEFAULT_TIME_GRID_FWD: Default forward time grid (0 to 1) for sampling.
        DEFAULT_TIME_GRID_BWD: Default backward time grid (1 to 0) for likelihood.
        CHECKPOINT_DIR_NAME: Subdirectory name for checkpoints.
        LOSS_PLOT_FILENAME: Filename for the training loss plot.
        DEFAULT_GRAD_ACCUM_STEPS: Default gradient accumulation steps.
        DEFAULT_AUTOCLIP_PERCENTILE: Default percentile for AutoClip.
        DEFAULT_AUTOCLIP_WINDOW_SIZE: Default window size for AutoClip.
    """

    # --- Default constants ---
    DEFAULT_SAVE_EVERY_EPOCHS: int = 1
    DEFAULT_NUM_STEPS: int = 7
    DEFAULT_TRAIN_WITH_MLE_NUM_STEPS: int = 5
    DEFAULT_METHOD: str = "heun3"
    DEFAULT_TRAIN_WITH_MLE_METHOD: str = 'euler'
    DEFAULT_ATOL: float = 1e-5
    DEFAULT_RTOL: float = 1e-5
    DEFAULT_TIME_GRID_FWD: Tensor = torch.tensor([0.0, 1.0])
    DEFAULT_TIME_GRID_BWD: Tensor = torch.tensor([1.0, 0.0])
    CHECKPOINT_DIR_NAME: str = 'checkpoints'
    LOSS_PLOT_FILENAME: str = 'training_loss_plot.png'
    DEFAULT_GRAD_ACCUM_STEPS: int = 1
    DEFAULT_AUTOCLIP_PERCENTILE: float = 10.0
    DEFAULT_AUTOCLIP_WINDOW_SIZE: int = 1000

    def __init__(
        self,
        save_dir_root: str,
        model_params: dict,
        model: nn.Module,
        model_wrapper: Type[Union[ModelWrapper, CFMWrapper,]],
        optimizer: optim.Optimizer,
        lr_scheduler: LRScheduler,
        loss_fn: nn.Module = nn.MSELoss(),
        training_path_scheduler: ConvexScheduler = CondOTScheduler(),
        sampling_path_scheduler: Optional[ConvexScheduler] = None,
        base_distribution: Optional[Distribution] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        autoclip_percentile: float = DEFAULT_AUTOCLIP_PERCENTILE,
        autoclip_window_size: int = DEFAULT_AUTOCLIP_WINDOW_SIZE,
    ) -> None:
        """Initializes the CFMExecutor.

        Args:
            save_dir_root: Root directory for saving checkpoints, plots, etc.
            model_params: Dictionary of model hyperparameters.
            model: The neural network model (e.g., a UNet).
            model_wrapper: A wrapper for the model, possibly adapting its
                interface for the ODESolver.
            optimizer: The PyTorch optimizer.
            lr_scheduler: The PyTorch learning rate scheduler.
            loss_fn: The loss function for CFM training (e.g., MSELoss).
            training_path_scheduler: Scheduler for defining the probability
                path during training.
            sampling_path_scheduler: Optional scheduler for defining the
                probability path during sampling. If None, training scheduler's
                path characteristics are assumed for the original model.
            base_distribution: An instantiated `torch.distributions.Distribution`
                object representing the base distribution $p_0$. Required for
                likelihood computation. If None, a warning is issued, and
                likelihood computation might default to a standard Normal.
            device: The computation device ('cuda' or 'cpu').
            autoclip_percentile: Percentile for AutoClip adaptive gradient
                clipping.
            autoclip_window_size: Window size for AutoClip.

        Raises:
            OSError: If the `save_dir_root` cannot be created.
            NotADirectoryError: If `save_dir_root` is not a directory.
            ValueError: If AutoClip parameters are invalid.
        """
        root = Path(save_dir_root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Save directory root '{root}' confirmed/created.")
        except OSError as e:
            logger.error(f"Error creating save directory root '{root}': {e}")
            raise

        if not root.is_dir():
            raise NotADirectoryError(
                f"Specified save directory root '{root}' is not a directory."
            )

        self.save_dir_root: Path = root
        self.model_params: dict = model_params
        self.device: str = device
        self.model: nn.Module = model.to(self.device)
        self.model_wrapper: ModelWrapper = model_wrapper
        self.optimizer: optim.Optimizer = optimizer
        self.lr_scheduler: LRScheduler = lr_scheduler
        self.loss_fn: nn.Module = loss_fn
        self.training_path_scheduler: ConvexScheduler = training_path_scheduler
        self.sampling_path_scheduler: Optional[ConvexScheduler] = sampling_path_scheduler
        self.base_distribution: Optional[Distribution] = base_distribution
        self.starting_epoch: int = 0
        self.epoch_losses: List[float] = []

        try:
            self.auto_clipper: AutoClip = AutoClip(
                model=self.model,
                percentile=autoclip_percentile,
                window_size=autoclip_window_size
            )
            logger.info(
                f"AutoClip initialized with percentile={autoclip_percentile}, "
                f"window_size={autoclip_window_size}."
            )
        except ValueError as e:
            logger.error(f"Error initializing AutoClip: {e}")
            raise

        logger.info(f"CFMExecutor initialized. Device: {self.device}")
        if torch.cuda.is_available() and self.device == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                logger.info(f"GPU Name: {gpu_name} - {gpu_mem} MB")
            except Exception as e:
                logger.warning(f"Could not get GPU details: {e}")

        logger.info(f"Save directory root: {self.save_dir_root}")

        if not _MATPLOTLIB_AVAILABLE:
            logger.warning(
                "Matplotlib not installed. Loss plotting will be disabled. "
                "Run 'pip install matplotlib' to enable."
            )
        if self.base_distribution is None:
            logger.warning(
                "Base distribution p0 not provided. Likelihood computation "
                "will default to standard Normal or may not be available if "
                "data shape is incompatible."
            )
        elif not isinstance(self.base_distribution, Distribution):
            logger.warning(
                f"Provided base_distribution is not an instance of "
                f"torch.distributions.Distribution (type: "
                f"{type(self.base_distribution)}). Likelihood computation "
                "might fail or behave unexpectedly."
            )

    def _setup_save_subdir(self, dir_name: str) -> Path:
        """Creates and returns a subdirectory within the root save directory.

        Args:
            dir_name: The name of the subdirectory to create.

        Returns:
            The Path object representing the created subdirectory.

        Raises:
            OSError: If the subdirectory cannot be created.
        """
        target_dir = self.save_dir_root / dir_name
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating directory {target_dir}: {e}")
            raise
        return target_dir

    def set_save_dir_root(self, dir_path: str) -> None:
        """Sets a new root directory for saving results.

        Args:
            dir_path: The path to the new root directory.

        Raises:
            OSError: If the new directory cannot be created.
            NotADirectoryError: If the new path is not a directory.
        """
        new_root = Path(dir_path)
        try:
            new_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"New save directory root '{new_root}' confirmed/created.")
        except OSError as e:
            logger.error(f"Error creating new save directory root '{new_root}': {e}")
            raise
        if not new_root.is_dir():
            raise NotADirectoryError(
                f"Newly set save directory root '{new_root}' is not a directory."
            )
        self.save_dir_root = new_root
        logger.info(f"Save directory root updated to: {self.save_dir_root}")

    def set_optimizer(self, optimizer: optim.Optimizer) -> None:
        """Updates the optimizer used by the executor.

        Args:
            optimizer: The new PyTorch optimizer instance.
        """
        self.optimizer = optimizer
        logger.info("Optimizer updated.")

    def set_lr_scheduler(self, lr_scheduler: LRScheduler) -> None:
        """Updates the learning rate scheduler used by the executor.

        Args:
            lr_scheduler: The new PyTorch LRScheduler instance.
        """
        self.lr_scheduler = lr_scheduler
        logger.info("Learning rate scheduler updated.")

    def set_training_path_scheduler(self, scheduler: ConvexScheduler) -> None:
        """Updates the training path scheduler.

        Args:
            scheduler: The new ConvexScheduler for training.
        """
        self.training_path_scheduler = scheduler
        logger.info("Training path scheduler updated.")

    def set_sampling_path_scheduler(self, scheduler: ConvexScheduler) -> None:
        """Updates the sampling path scheduler.

        Args:
            scheduler: The new ConvexScheduler for sampling.
        """
        self.sampling_path_scheduler = scheduler
        logger.info("Sampling path scheduler updated.")

    def set_base_distribution(self, distribution: Distribution) -> None:
        """Updates the base distribution $p_0$.

        Args:
            distribution: The new `torch.distributions.Distribution` instance.
        """
        if not isinstance(distribution, Distribution):
            logger.warning(
                f"Setting base distribution to a non-torch.distributions.Distribution "
                f"object: {type(distribution)}. This may cause issues."
            )
        self.base_distribution = distribution
        logger.info(f"Base distribution updated to: {type(distribution)}")

    # --- Core Training Logic ---
    def _train_step(
        self,
        data: Tensor,
        label: Tensor,
        prob_path: AffineProbPath,
        gradient_accumulation_steps: int = 1,
        kinetic_regularization: float = None,
        mu_logit: float = 0.0,
        sigma_logit: float = 1.0
    ) -> float:
        """Performs a single training step (forward pass and loss calculation).

        This involves sampling points along the probability path, computing the
        model's predicted velocity, calculating the loss against the true
        velocity, and scaling the loss for gradient accumulation.

        Args:
            data: The input data batch ($x_1$).
            label: Corresponding labels or conditions for the data.
            prob_path: The probability path generator.
            gradient_accumulation_steps: The number of steps over which
                gradients are accumulated. The loss is scaled by this factor.
            kinetic_regularization: Optional regularization term for kinetic energy.
            mu_logit: Mean for the logit-normal sampler.
            sigma_logit: Standard deviation for the logit-normal sampler.

        Returns:
            The unscaled loss value for this step.
        """
        if isinstance(label, Tensor):
            x1, conds = data.to(self.device), label.to(self.device)
            multi_label = False
        elif isinstance(label, list):
            x1, conds = data.to(self.device), [c.to(self.device) for c in label]
            multi_label = True
        else:
            raise TypeError(
                "Label must be a Tensor or a list of Tensors. "
                f"Received type: {type(label)}"
            )

        # Sample $x_0$ from a standard Normal distribution (common choice for base).
        x_0 = torch.randn_like(x1, device=self.device)

        # Sample time $t$ using a logit-normal sampler, typically biasing
        # samples towards the ends of the [0,1] interval.
        t = logit_normal_sampler(
            shape=(x1.shape[0],),
            mu_logit=mu_logit,
            sigma_logit=sigma_logit,
            device=self.device
        )

        # Sample $x_t$ and $dx_t/dt$ (target velocity) from the probability path.
        path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x1)

        # Get model's predicted velocity $v(x_t, t, condition)$.
        model_output = self.model(
            x=path_sample.x_t, time=path_sample.t,
            conditions=[conds,] if not multi_label else conds
        )

        # Compute loss (e.g., MSE between predicted and target velocity).
        loss = self.loss_fn(model_output, path_sample.dx_t)

        if kinetic_regularization is not None:
            kinetic_loss = torch.pow(model_output, 2).mean() * kinetic_regularization
            loss += kinetic_loss

        # Scale loss for gradient accumulation.
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        return loss.item()

    def train(
        self,
        train_loaders: Tuple[DataLoader, ...],
        num_epochs: int,
        gradient_accumulation_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
        save_every_epochs: int = DEFAULT_SAVE_EVERY_EPOCHS,
        kinetic_regularization: float = None,
        mu_logit: float = 0.0,
        sigma_logit: float = 1.0,
        sigma_logit_increase_per_epoch: float = 0.0,
        data_argumentation_fn: Callable[[Tensor], Tensor] = None,
        use_amp: bool = False,
    ) -> None:
        """Runs the main training loop.

        This method iterates over epochs and batches, performing training steps,
        accumulating gradients, applying adaptive gradient clipping (AutoClip),
        updating model weights, and adjusting the learning rate. It logs
        progress, saves checkpoints, and plots the loss curve upon completion.

        Args:
            train_loaders: A tuple of DataLoaders providing training data batches.
                Batches are drawn randomly from these loaders.
            num_epochs: The total number of epochs to train for.
            gradient_accumulation_steps: Number of batches to accumulate
                gradients over before performing an optimizer step.
            save_every_epochs: Frequency (in epochs) for saving checkpoints.
            kinetic_regularization: Optional regularization term for kinetic energy.
            mu_logit: Initial value for the logit-normal sampler's mean.
            sigma_logit: Initial value for the logit-normal sampler's standard deviation.
            sigma_logit_increase_per_epoch: Amount to increase the `sigma_logit`
                after each epoch. This can help explore the time distribution
                more effectively as training progresses.
            data_argumentation_fn: Optional function to apply data argumentation
                to the input data batch. This function should take a Tensor and
                return a modified Tensor. If None, no argumentation is applied.
            use_amp: Whether to use Automatic Mixed Precision (AMP) for training.

        Raises:
            ValueError: If `gradient_accumulation_steps`, `num_epochs`, or
                `save_every_epochs` are not positive integers, or if
                `train_loaders` is empty or contains DataLoaders that sum to
                zero total batches.
            TypeError: If any DataLoader in `train_loaders` does not support `len()`.
        """
        # --- Input Validation ---
        if not isinstance(gradient_accumulation_steps, int) or \
           gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        if not train_loaders:
            raise ValueError("train_loaders tuple cannot be empty.")
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(save_every_epochs, int) or save_every_epochs <= 0:
            raise ValueError("save_every_epochs must be a positive integer.")

        prob_path = AffineProbPath(scheduler=self.training_path_scheduler)
        self.model.train() # Set model to training mode

        try:
            total_batches_per_epoch = sum(len(dl) for dl in train_loaders)
            if total_batches_per_epoch == 0:
                raise ValueError(
                    "Total number of batches across all DataLoaders is zero. "
                    "Check DataLoader configurations."
                )
            logger.info(f"Total batches per epoch from all loaders: {total_batches_per_epoch}")
        except TypeError: # pragma: no cover
            # This typically occurs if a DataLoader is an IterableDataset without __len__
            logger.error(
                "One or more DataLoaders do not support len(). Cannot determine "
                "total batches per epoch for epoch-based training with multi_dataloader_random_cycle."
            )
            raise TypeError(
                "All DataLoaders must support len() for this training setup."
            )

        # Estimate effective batch size for logging purposes
        first_loader_batch_size = getattr(train_loaders[0], 'batch_size', None)
        if first_loader_batch_size:
            effective_batch_size = first_loader_batch_size * gradient_accumulation_steps
            logger.info(
                f"Effective batch size (approx, based on first loader): {effective_batch_size}"
            )
        else: # pragma: no cover
            logger.warning(
                "Could not infer batch size from the first DataLoader. "
                "Effective batch size information will be unavailable."
            )

        logger.info(f"Starting training for {num_epochs} epochs.")
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

        self.optimizer.zero_grad() # Clear any existing gradients

        # Initialize epoch losses list if starting fresh
        if self.starting_epoch == 0:
            self.epoch_losses = []

        logger.info(
            f"Using logit-normal sampler with mu_logit={mu_logit:.4f},"
            f" sigma_logit={sigma_logit:.4f}. "
            "This controls the sampling distribution for time t."
        )

        if use_amp:
            logger.info(f"Using AMP device type: {self.device}")

        # --- Main Training Loop ---
        with torch.autocast(device_type=self.device, enabled=use_amp):
            for epoch in range(self.starting_epoch, num_epochs):
                epoch_start_time = time.time()
                self.model.train() # Ensure model is in training mode each epoch
                running_epoch_loss = 0.0
                processed_batches_in_epoch = 0
                processed_batches_in_accum_cycle = 0
                optimizer_steps_in_epoch = 0
                last_clip_value_used = None
                if epoch > self.starting_epoch: # Increase sigma_logit after the first epoch
                    if sigma_logit_increase_per_epoch != 0:
                        sigma_logit += sigma_logit_increase_per_epoch
                        logger.info(
                            f"Increasing sigma_logit by {sigma_logit_increase_per_epoch:.4f} "
                            f"for this epoch. New sigma_logit: {sigma_logit:.4f}"
                        )
                    else:
                        logger.info(
                            f"Keeping sigma_logit = {sigma_logit:.4f} for this epoch."
                        )

                # Create a batch iterator that randomly cycles through dataloaders
                batch_iterator = multi_dataloader_random_cycle(
                    train_loaders, total_batches_per_epoch
                )
                progress_bar = tqdm(
                    batch_iterator,
                    total=total_batches_per_epoch,
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                    leave=False # Keep the bar nested under epoch logs
                )

                for batch_idx, batch_content in enumerate(progress_bar):
                    # Unpack data and label, providing a default label if not present
                    if isinstance(batch_content, (list, tuple)) and len(batch_content) == 2:
                        data_batch, label_batch = batch_content
                    else:
                        data_batch = batch_content
                        # Default label (e.g., for unconditional models or if labels are implicit)
                        label_batch = torch.zeros(data_batch.shape[0], dtype=torch.long)

                    if data_argumentation_fn is not None:
                        # Apply data argumentation function if provided
                        data_batch = data_argumentation_fn(data_batch)

                    step_loss = self._train_step(
                        data_batch,
                        label_batch,
                        prob_path,
                        gradient_accumulation_steps,
                        kinetic_regularization=kinetic_regularization,
                        mu_logit=mu_logit,
                        sigma_logit=sigma_logit
                    )
                    running_epoch_loss += step_loss
                    processed_batches_in_epoch += 1
                    processed_batches_in_accum_cycle += 1

                    # Perform optimizer step after accumulating gradients or if it's the last batch
                    is_last_batch_in_epoch = (batch_idx + 1) == total_batches_per_epoch
                    if (processed_batches_in_accum_cycle == gradient_accumulation_steps or
                            is_last_batch_in_epoch):

                        # Apply adaptive gradient clipping before the optimizer step
                        clip_val = self.auto_clipper()
                        if clip_val is not None:
                            last_clip_value_used = clip_val

                        self.optimizer.step()
                        self.optimizer.zero_grad() # Reset gradients for the next accumulation cycle
                        optimizer_steps_in_epoch += 1
                        processed_batches_in_accum_cycle = 0 # Reset accumulation counter

                    # Update progress bar postfix with current stats
                    if (batch_idx + 1) % 10 == 0 or is_last_batch_in_epoch:
                        current_avg_loss = running_epoch_loss / processed_batches_in_epoch \
                                           if processed_batches_in_epoch > 0 else 0.0
                        postfix_str = (
                            f'Avg Loss: {current_avg_loss:.4f}, '
                            f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                        )
                        if last_clip_value_used is not None:
                            postfix_str += f', Clip: {last_clip_value_used:.2f}'
                        progress_bar.set_postfix_str(postfix_str)

                # --- End of Epoch ---
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                average_epoch_loss = running_epoch_loss / total_batches_per_epoch \
                                     if total_batches_per_epoch > 0 else 0.0
                self.epoch_losses.append(average_epoch_loss)

                logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f}s.")
                logger.info(f"  Average Epoch Loss: {average_epoch_loss:.4f}")
                logger.info(f"  Optimizer steps in epoch: {optimizer_steps_in_epoch}")
                if last_clip_value_used is not None:
                    logger.info(
                        f"  Last AutoClip threshold used: {last_clip_value_used:.4f}"
                    )

                self.lr_scheduler.step() # Step the learning rate scheduler
                logger.info(
                    f"  LR scheduler stepped. New LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

                # Save checkpoint periodically or at the end of training
                if (epoch + 1) % save_every_epochs == 0 or (epoch + 1) == num_epochs:
                    autoclipper_state = self.auto_clipper.grad_history \
                                        if hasattr(self, 'auto_clipper') else None
                    self.save_checkpoint(
                        epoch=epoch + 1, # Epochs are 1-indexed for saving
                        loss_history=self.epoch_losses,
                        autoclipper_state=autoclipper_state
                    )
                logger.info("-" * 60) # Separator for readability

        logger.info("Training finished.")
        self._plot_and_save_loss(plot_filename=self.LOSS_PLOT_FILENAME)

    def _plot_and_save_loss(self, plot_filename: str) -> None:
        """Plots the average training loss per epoch and saves the plot.

        Args:
            plot_filename: The filename for saving the loss plot (e.g.,
                "training_loss.png"). The plot is saved in `self.save_dir_root`.
        """
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib is unavailable. Skipping loss plotting.")
            return
        if not self.epoch_losses:
            logger.info("No epoch loss data recorded; skipping plotting.")
            return

        try:
            plt.figure(figsize=(10, 6))
            # If self.epoch_losses contains all losses from epoch 1:
            epochs_to_plot = range(1, len(self.epoch_losses) + 1)


            plt.plot(
                epochs_to_plot, self.epoch_losses, marker='o',
                linestyle='-', label='Average Loss per Epoch'
            )
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True)

            # Adjust x-axis ticks for better readability
            if len(list(epochs_to_plot)) < 20: # Show all epoch numbers if few epochs
                plt.xticks(list(epochs_to_plot))
            else: # Otherwise, let matplotlib decide tick locations
                plt.locator_params(axis='x', integer=True, nbins='auto')

            plt.tight_layout()
            save_path = self.save_dir_root / plot_filename
            plt.savefig(save_path)
            plt.close() # Close the figure to free memory
            logger.info(f"Training loss plot saved to: {save_path}")
        except Exception as e: # pragma: no cover
            logger.error(f"Error plotting or saving loss curve: {e}")

    def compute_likelihood(
        self,
        likelihood_loader: DataLoader,
        num_samples: int,
        num_steps: int = DEFAULT_NUM_STEPS,
        exact_divergence: bool = False,
        method: str = DEFAULT_METHOD,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        time_grid: Tensor = DEFAULT_TIME_GRID_BWD,
        timestep_schedule_mu: float = -1.1,
        timestep_schedule_sigma: float = 1.0,
        do_log: bool = True,
        reshape: bool = True,
        **model_extras
    ) -> Optional[Tensor]:
        """Computes log-likelihood of data samples using the instantaneous change of variables formula.

        This method evaluates $p_1(x_1)$ by solving the reverse ODE from $x_1$
        to $x_0$ and accumulating the divergence of the vector field.
        The log-likelihood is given by:
        $log p_1(x_1) = log p_0(x_0) - \int_0^1 div(v(x_t, t)) dt$.
        The integral term is the change in log density.

        Args:
            likelihood_loader: DataLoader providing target samples ($x_1$) and
                optionally conditions.
            num_samples: Expected number of samples corresponding to each original
                data item if data was augmented/repeated for the loader. Used for
                reshaping the output if `reshape` is True.
            num_steps: Number of steps for fixed-step ODE solvers.
            exact_divergence: Whether to use exact divergence computation (if
                available) or Hutchinson's estimator.
            method: ODE solver method (e.g., 'heun3', 'dopri5').
            atol: Absolute tolerance for adaptive ODE solvers.
            rtol: Relative tolerance for adaptive ODE solvers.
            time_grid: Time grid for the reverse ODE solve (typically from 1.0 to 0.0).
            do_log: Whether to log progress and results.
            reshape: If True, attempts to reshape the output tensor of
                likelihoods based on `num_samples` to group likelihoods
                belonging to the same original data item.
            **model_extras: Additional keyword arguments passed to the model
                during velocity field evaluation (e.g., `conditions`).

        Returns:
            A Tensor containing the log-likelihood (in nats) for each sample.
            The shape might be `(num_original_items, num_samples)` if `reshape`
            is True and successful, otherwise `(total_processed_samples,)`.
            Returns `None` if likelihood computation fails or no samples
            are processed.
        """
        first_batch_x1_shape = None # For determining L if base_distribution is None
        # Peek at the first batch to get shape for default base_distribution
        try:
            peek_batch = next(iter(likelihood_loader))
            if isinstance(peek_batch, (list, tuple)):
                first_batch_x1_shape = peek_batch[0].shape
            else:
                first_batch_x1_shape = peek_batch.shape
        except StopIteration:
             logger.error("Likelihood loader is empty. Cannot compute likelihoods.")
             return None
        except Exception as e:
            logger.warning(f"Could not peek into likelihood_loader to get shape: {e}")
            # Proceed, but default base distribution might fail if shape is not (..., L, L)

        if self.base_distribution is None:
            if first_batch_x1_shape and len(first_batch_x1_shape) >= 2: # Assuming at least (B, ..., L, L)
                L = first_batch_x1_shape[-1] # Assume square, takes last dim
                # This default is quite specific and assumes image-like data.
                # Users should ideally provide a base_distribution.
                self.base_distribution = Independent(
                    Normal(
                        loc=torch.zeros(1, 1, L, L, device=self.device),
                        scale=torch.ones(1, 1, L, L, device=self.device)
                    ),
                    reinterpreted_batch_ndims=0
                )
                logger.info(
                    f"Base distribution p0 not provided. Defaulting to standard "
                    f"Normal(0,1))."
                )
            else:
                logger.error(
                    "Base distribution p0 not provided and could not infer "
                    "data dimensions (e.g., for images L,L) to create a default. "
                    "Likelihood computation cannot proceed."
                )
                return None


        if not isinstance(self.base_distribution, Distribution):
            logger.error(
                f"Cannot compute likelihood: base_distribution is not a "
                f"torch.distributions.Distribution (type: {type(self.base_distribution)})."
            )
            return None
        if not callable(getattr(self.base_distribution, 'log_prob', None)):
            logger.error(
                "Cannot compute likelihood: base_distribution does not have a "
                "callable 'log_prob' method."
            )
            return None

        self.model.eval() # Set model to evaluation mode
        solver = ODESolver(velocity_model=self.model_wrapper(self.model))

        all_likelihoods_list: List[Tensor] = []
        total_samples_processed = 0
        # Capture shape of x_1 from first processed batch for BPD calculation later
        actual_first_batch_x1_shape = None

        if do_log:
            logger.info("Starting likelihood computation...")
            logger.info(f"  Solver: {method}, Time Grid (Original): {time_grid.tolist()}")
            if method in ['euler', 'midpoint', 'rk4', 'heun2', 'heun3']: # Fixed-step solvers
                # Create a linearly spaced time grid for fixed-step solvers
                time_grid_ode = torch.linspace(
                    time_grid[0].item(), time_grid[-1].item(),
                    steps=num_steps + 1, device=self.device
                )
                logger.info(f"  Time Grid (Fixed Steps): {time_grid_ode.tolist()}")
            else: # Adaptive solvers use atol, rtol
                time_grid_ode = time_grid.to(self.device) # Use original grid for adaptive
                logger.info(f"  Tolerances for adaptive solver: atol={atol}, rtol={rtol}")

        # Define a helper for log p0(x) that sums over event dimensions
        def log_p0_evaluation_func(x_at_t0: Tensor) -> Tensor:
            """Calculates log p0(x_at_t0), summing over event dimensions."""
            # log_prob_elementwise might have shape [batch_size, *event_shape]
            # or just [batch_size] if event_shape is ()
            log_prob_val = self.base_distribution.log_prob(x_at_t0)

            # Sum over all dimensions except the batch dimension (dim 0)
            # to get a single scalar log-probability per item in the batch.
            reduce_dims = list(range(1, x_at_t0.dim()))
            if not reduce_dims: # If x_at_t0 is 1D [batch_size]
                return log_prob_val
            else:
                return log_prob_val.sum(dim=reduce_dims)

        # Apply timestep scheduling (e.g., variance preserving schedule) to the time grid
        # 'a=3' is a hyperparameter for this specific scheduler
        time_grid_scheduled_for_ode = timestep_scheduler(
            time_grid_ode,
            mu=timestep_schedule_mu,
            sigma=timestep_schedule_sigma,
        )
        if do_log:
            logger.info(f"  Time Grid (Scheduled for ODE): {time_grid_scheduled_for_ode.tolist()}")

        with torch.no_grad(): # Likelihood computation is an evaluation pass
            progress_bar = tqdm(
                likelihood_loader, desc="Computing Likelihood",
                leave=False, total=len(likelihood_loader)
            )
            for batch_content in progress_bar:
                current_model_extras = model_extras.copy()
                if isinstance(batch_content, (list, tuple)):
                    x_1_batch = batch_content[0].to(self.device)
                    if len(batch_content) > 1:
                        conditions_batch = batch_content[1].to(self.device)
                        current_model_extras['conditions'] = [conditions_batch, ]
                    if actual_first_batch_x1_shape is None:
                        actual_first_batch_x1_shape = x_1_batch.shape
                else:
                    x_1_batch = batch_content.to(self.device)
                    current_model_extras.pop('conditions', None) # Clear if not provided
                    if actual_first_batch_x1_shape is None:
                        actual_first_batch_x1_shape = x_1_batch.shape

                batch_size = x_1_batch.shape[0]

                try:
                    # compute_likelihood returns (x0, log_p1_batch)
                    _, log_p1_for_batch = solver.compute_likelihood(
                        x_1=x_1_batch,
                        log_p0=log_p0_evaluation_func,
                        step_size=None, # Handled by num_steps for fixed, or adaptive
                        method=method,
                        atol=atol,
                        rtol=rtol,
                        time_grid=time_grid_scheduled_for_ode,
                        return_intermediates=False,
                        exact_divergence=exact_divergence,
                        enable_grad=False, # No gradients needed for evaluation
                        **current_model_extras
                    )

                    if torch.isnan(log_p1_for_batch).any() or \
                       torch.isinf(log_p1_for_batch).any():
                        logger.warning(
                            "NaN or Inf detected in log-likelihood for a batch. "
                            "Skipping this batch."
                        )
                        continue

                    all_likelihoods_list.append(log_p1_for_batch.cpu()) # Store on CPU
                    total_samples_processed += batch_size

                except Exception as e: # pragma: no cover
                    logger.error(
                        f"Error during likelihood computation for a batch: {e}",
                        exc_info=True # Provides traceback
                    )
                    logger.warning("Skipping affected batch due to error.")
                    continue

        if not all_likelihoods_list:
            logger.error("Likelihood computation failed: No likelihoods were collected.")
            return None

        # Concatenate all collected likelihoods
        concatenated_likelihoods = torch.cat(all_likelihoods_list, dim=0)
        final_likelihood_tensor = concatenated_likelihoods # Default if reshape fails or is false

        # --- Reshape Logic (if requested) ---
        if reshape:
            try:
                # Try to get dataset size if loader is from a standard Dataset
                dataset_size = len(likelihood_loader.dataset) \
                               if hasattr(likelihood_loader, 'dataset') and \
                                  likelihood_loader.dataset is not None else None

                actual_items_processed_total = concatenated_likelihoods.shape[0]

                # Determine the number of "original" items before any duplication/augmentation
                # that `num_samples` is intended to account for.
                if num_samples <= 0:
                     logger.warning(
                         f"num_samples ({num_samples}) is not positive. "
                         "Cannot perform meaningful reshape. Returning concatenated likelihoods."
                     )
                     num_original_items = actual_items_processed_total # Treat each as unique
                     effective_num_samples_per_item = 1
                elif dataset_size is not None:
                    if dataset_size % num_samples != 0:
                        logger.warning(
                            f"Dataset size ({dataset_size}) is not perfectly "
                            f"divisible by num_samples ({num_samples}). Reshape might "
                            "group items unexpectedly."
                        )
                    num_original_items = dataset_size // num_samples
                    effective_num_samples_per_item = num_samples
                    if actual_items_processed_total != dataset_size:
                        logger.warning(
                            f"Total processed items ({actual_items_processed_total}) "
                            f"does not match dataset size ({dataset_size}). This could "
                            "indicate an issue or data loader dropping last batch. "
                            "Reshape will be based on actual processed items."
                        )
                        # Adjust num_original_items based on what was actually processed
                        num_original_items = actual_items_processed_total // num_samples
                else: # Dataset size unknown, estimate from processed items
                    if actual_items_processed_total % num_samples != 0:
                        logger.warning(
                            f"Total processed items ({actual_items_processed_total}) "
                            f"is not perfectly divisible by num_samples ({num_samples}). "
                            "Reshape might be inexact."
                        )
                    num_original_items = actual_items_processed_total // num_samples
                    effective_num_samples_per_item = num_samples


                # Perform reshape if conditions are met
                if num_original_items > 0 and \
                   actual_items_processed_total == num_original_items * effective_num_samples_per_item :
                    # Target shape: [num_original_items, num_samples_per_item]
                    # (assuming likelihoods are scalar per sample)
                    final_likelihood_tensor = concatenated_likelihoods.reshape(
                        num_original_items, effective_num_samples_per_item
                    )
                    if do_log:
                        logger.info(
                            f"Reshaped likelihoods tensor shape: {final_likelihood_tensor.shape}"
                        )
                else:
                    logger.warning(
                        f"Cannot reshape likelihoods as expected (num_original_items={num_original_items}, "
                        f"num_samples_per_item={effective_num_samples_per_item}, actual_processed={actual_items_processed_total}). "
                        f"Returning concatenated likelihoods with shape: {concatenated_likelihoods.shape}"
                    )
                    # final_likelihood_tensor remains concatenated_likelihoods
            except Exception as e: # pragma: no cover
                logger.error(
                    f"Error during likelihood reshaping. Concatenated shape: "
                    f"{concatenated_likelihoods.shape}. Error: {e}. "
                    "Returning concatenated likelihoods."
                )
                # final_likelihood_tensor remains concatenated_likelihoods
        else: # reshape is False
            if do_log:
                logger.info(
                    f"Concatenated likelihoods shape (reshape=False): "
                    f"{final_likelihood_tensor.shape}"
                )
        # --- End Reshape Logic ---

        # --- Log Average Log-Likelihood and BPD (for informational purposes) ---
        if total_samples_processed > 0:
            # Use the original concatenated tensor for overall average calculation
            average_log_likelihood = concatenated_likelihoods.sum().item() / total_samples_processed
            if do_log:
                logger.info("Likelihood computation finished.")
                logger.info(f"  Processed {total_samples_processed} samples in total.")
                logger.info(
                    f"  Average Log-Likelihood (nats, for info only): {average_log_likelihood:.4f}"
                )

                if actual_first_batch_x1_shape is not None and \
                   len(actual_first_batch_x1_shape) > 1: # Need at least [B, D1, ...]
                    try:
                        # Calculate data dimensionality (excluding batch dimension)
                        data_dim = float(np.prod(actual_first_batch_x1_shape[1:]))
                        if data_dim > 0:
                             # Bits Per Dimension (BPD) = -log2(p(x)) / num_dimensions
                             # = -log_e(p(x)) / (num_dimensions * log_e(2))
                            bpd = -average_log_likelihood / (data_dim * np.log(2.0))
                            logger.info(
                                f"  Average Bits Per Dimension (BPD, for info only): {bpd:.4f}"
                            )
                        else: # pragma: no cover
                            logger.warning("Data dimensionality is zero, cannot calculate BPD.")
                    except Exception as e: # pragma: no cover
                        logger.warning(f"Could not calculate BPD: {e}")
                else: # pragma: no cover
                    logger.warning(
                        "Could not determine data dimensionality from first batch "
                        "for BPD calculation."
                    )
        else: # Should have been caught by `if not all_likelihoods_list:`
             logger.warning(
                 "Total samples processed is zero. Cannot compute average log-likelihood."
             )

        return final_likelihood_tensor

    def train_with_maximum_likelihood(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        gradient_accumulation_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
        save_every_epochs: int = DEFAULT_SAVE_EVERY_EPOCHS,
        balance: float = 0.5,
        exact_divergence: bool = False,
        num_steps: int = DEFAULT_TRAIN_WITH_MLE_NUM_STEPS,
        timestep_schedule_mu: float = -1.1,
        timestep_schedule_sigma: float = 1.0,
        method: str = DEFAULT_TRAIN_WITH_MLE_METHOD,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        time_grid: Tensor = DEFAULT_TIME_GRID_BWD,
    ) -> None:
        """Trains the model using a combination of CFM loss and Maximum Likelihood Estimation (MLE).

        This training method augments the standard CFM objective (matching
        the vector field of a probability path) with an MLE objective. The MLE
        part involves computing the log-likelihood of the data and maximizing it.
        The final loss is a weighted sum of the CFM loss and the negative
        log-likelihood.

        Args:
            train_loader: DataLoader providing training data ($x_1$) and
                optionally conditions.
            num_epochs: Total number of epochs for training.
            gradient_accumulation_steps: Number of batches to accumulate
                gradients over before an optimizer step.
            save_every_epochs: Frequency (in epochs) for saving checkpoints.
            balance: Weight factor (0 to 1) for the MLE loss component.
                Loss = (1-balance) * CFM_Loss + balance * Scale * (-LogLikelihood).
            exact_divergence: Whether to use exact divergence for likelihood
                computation or Hutchinson's estimator.
            num_steps: Number of steps for fixed-step ODE solvers during
                likelihood computation within the training loop.
            method: ODE solver method for likelihood computation.
            atol: Absolute tolerance for adaptive ODE solvers (for likelihood).
            rtol: Relative tolerance for adaptive ODE solvers (for likelihood).
            time_grid: Time grid for reverse ODE solve in likelihood computation
                (typically 1.0 to 0.0).

        Raises:
            ValueError: If input parameters like `num_epochs`, `gradient_accumulation_steps`,
                        or `save_every_epochs` are invalid. Or if `train_loader` is empty.
            RuntimeError: If `base_distribution` is not set up correctly for likelihood.
        """
        # --- Input Validation and Setup for Base Distribution ---
        try:
            peek_batch = next(iter(train_loader))
            if isinstance(peek_batch, (list, tuple)):
                first_batch_x1_shape = peek_batch[0].shape
            else:
                first_batch_x1_shape = peek_batch.shape
        except StopIteration:
             logger.error("Train loader is empty. Cannot start MLE training.")
             raise ValueError("train_loader cannot be empty for MLE training.")


        if self.base_distribution is None:
            if first_batch_x1_shape and len(first_batch_x1_shape) >= 2:
                L = first_batch_x1_shape[-1]
                self.base_distribution = Independent(
                    Normal(
                        loc=torch.zeros(1, 1, L, L, device=self.device),
                        scale=torch.ones(1, 1, L, L, device=self.device)
                    ),
                    reinterpreted_batch_ndims=0
                )
                logger.info(
                    f"Base distribution p0 not provided for MLE training. Defaulting to "
                    f"standard Normal(0,1))."
                )
            else:
                err_msg = (
                    "Base distribution p0 not provided and could not infer data "
                    "dimensions for default. MLE training requires a valid base_distribution."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)


        if not isinstance(self.base_distribution, Distribution):
            raise RuntimeError(
                f"Cannot compute likelihood for MLE: base_distribution is not a "
                f"torch.distributions.Distribution (type: {type(self.base_distribution)})."
            )
        if not callable(getattr(self.base_distribution, 'log_prob', None)):
            raise RuntimeError(
                "Cannot compute likelihood for MLE: base_distribution does not "
                "have a callable 'log_prob' method."
            )

        if not isinstance(gradient_accumulation_steps, int) or \
           gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        # train_loader emptiness is checked by peeking.
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(save_every_epochs, int) or save_every_epochs <= 0:
            raise ValueError("save_every_epochs must be a positive integer.")
        if not 0.0 <= balance <= 1.0:
            logger.warning(f"Balance factor {balance} is outside [0,1]. Clipping or check logic.")
            balance = max(0.0, min(1.0, balance))


        # --- Likelihood-specific Setup ---
        def log_p0_mle_func(x_at_t0: Tensor) -> Tensor:
            """Calculates log p0(x_at_t0) for MLE, summing over event dimensions."""
            log_prob_val = self.base_distribution.log_prob(x_at_t0)
            reduce_dims = list(range(1, x_at_t0.dim()))
            return log_prob_val.sum(dim=reduce_dims) if reduce_dims else log_prob_val

        logger.info(f"MLE Training Solver: {method}, Time Grid (Original): {time_grid.tolist()}")
        if method in ['euler', 'midpoint', 'rk4', 'heun2', 'heun3']: # Fixed-step
            time_grid_mle_ode = torch.linspace(
                time_grid[0].item(), time_grid[-1].item(),
                steps=num_steps + 1, device=self.device
            )
            logger.info(f"  Time Grid for MLE (Fixed Steps): {time_grid_mle_ode.tolist()}")
        else: # Adaptive
            time_grid_mle_ode = time_grid.to(self.device)
            logger.info(f"  Tolerances for MLE adaptive solver: atol={atol}, rtol={rtol}")

        time_grid_mle_scheduled = timestep_scheduler(
            time_grid_mle_ode,
            mu=timestep_schedule_mu,
            sigma=timestep_schedule_sigma,
        )
        logger.info(f"  Time Grid for MLE (Scheduled for ODE): {time_grid_mle_scheduled.tolist()}")

        # --- Standard CFM Training Setup ---
        prob_path = AffineProbPath(scheduler=self.training_path_scheduler)
        self.model.train()
        # ODESolver for likelihood computation within the training loop
        likelihood_solver = ODESolver(velocity_model=self.model_wrapper(self.model))

        loader_batch_size = getattr(train_loader, 'batch_size', None)
        if loader_batch_size:
            effective_batch_size = loader_batch_size * gradient_accumulation_steps
            logger.info(f"Effective batch size for MLE training: {effective_batch_size}")
        else: # pragma: no cover
            logger.warning(
                "Could not infer batch size from DataLoader for MLE training. "
                "Effective batch size unknown."
            )

        logger.info(f"Starting MLE training for {num_epochs} epochs.")
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        logger.info(f"Loss balance (CFM vs -LogLike): {1-balance:.2f} vs {balance:.2f}")


        self.optimizer.zero_grad()
        if self.starting_epoch == 0:
            self.epoch_losses = []

        # --- Main MLE Training Loop ---
        for epoch in range(self.starting_epoch, num_epochs):
            epoch_start_time = time.time()
            self.model.train() # Ensure model is in training mode
            running_epoch_loss = 0.0
            processed_batches_in_epoch = 0
            processed_batches_in_accum_cycle = 0
            optimizer_steps_in_epoch = 0
            last_clip_value_used = None

            progress_bar = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"MLE Epoch {epoch + 1}/{num_epochs}",
                leave=False
            )

            for batch_idx, batch_content in enumerate(progress_bar):
                current_model_extras = {} # For conditions in likelihood
                if isinstance(batch_content, (list, tuple)) and len(batch_content) == 2:
                    data_batch, label_batch = batch_content
                    condition_batch = label_batch.to(self.device)
                    current_model_extras['conditions'] = [condition_batch, ]
                else:
                    data_batch = batch_content
                    # Default label for CFM part; likelihood part might use implicit conditions
                    label_batch = torch.zeros(data_batch.shape[0], dtype=torch.long)
                    # No explicit conditions for likelihood solver if not provided

                x1_batch = data_batch.to(self.device)

                # --- CFM Loss Component ---
                x0_cfm = torch.randn_like(x1_batch, device=self.device)
                t_cfm = logit_normal_sampler(shape=(x1_batch.shape[0],), device=self.device)
                path_sample_cfm = prob_path.sample(t=t_cfm, x_0=x0_cfm, x_1=x1_batch)
                model_output_cfm = self.model(
                    x=path_sample_cfm.x_t,
                    time=path_sample_cfm.t,
                    conditions=[label_batch.to(self.device), ] # Use label_batch for CFM conditions
                )
                loss_cfm = self.loss_fn(model_output_cfm, path_sample_cfm.dx_t)

                # --- MLE Loss Component ---
                # Gradients must flow through likelihood computation for MLE objective
                # Therefore, enable_grad=True for the likelihood_solver call.
                _, log_p1_batch_mle = likelihood_solver.compute_likelihood(
                    x_1=x1_batch,
                    log_p0=log_p0_mle_func,
                    step_size=None,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    time_grid=time_grid_mle_scheduled,
                    return_intermediates=False,
                    exact_divergence=exact_divergence,
                    enable_grad=True, # CRITICAL: Enable gradients for MLE
                    **current_model_extras # Pass conditions if available
                )
                # MLE aims to maximize log_likelihood, so loss is -log_likelihood.
                # Mean over batch for a scalar loss component.
                loss_mle = -log_p1_batch_mle.mean()

                # Combine losses
                # A small multiplier (e.g., 1e-1) for MLE loss can sometimes stabilize training,
                # depending on the relative scales of CFM and likelihood losses.
                combined_loss = (1 - balance) * loss_cfm + balance * 1e-1 * loss_mle

                scaled_loss = combined_loss / gradient_accumulation_steps
                scaled_loss.backward() # Accumulate gradients

                running_epoch_loss += combined_loss.item()
                processed_batches_in_epoch += 1
                processed_batches_in_accum_cycle += 1

                is_last_batch_in_epoch = (batch_idx + 1) == len(train_loader)
                if processed_batches_in_accum_cycle == gradient_accumulation_steps or \
                   is_last_batch_in_epoch:
                    clip_val = self.auto_clipper()
                    if clip_val is not None:
                        last_clip_value_used = clip_val

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    optimizer_steps_in_epoch += 1
                    processed_batches_in_accum_cycle = 0

                if (batch_idx + 1) % 10 == 0 or is_last_batch_in_epoch:
                    current_avg_loss = running_epoch_loss / processed_batches_in_epoch \
                                       if processed_batches_in_epoch > 0 else 0.0
                    postfix_str = (
                        f'Avg Loss: {current_avg_loss:.4f}, '
                        f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                    )
                    if last_clip_value_used is not None:
                        postfix_str += f', Clip: {last_clip_value_used:.2f}'
                    progress_bar.set_postfix_str(postfix_str)

            # --- End of MLE Epoch ---
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            average_epoch_loss = running_epoch_loss / len(train_loader) \
                                 if len(train_loader) > 0 else 0.0
            self.epoch_losses.append(average_epoch_loss)

            logger.info(f"MLE Epoch {epoch + 1} completed in {epoch_duration:.2f}s.")
            logger.info(f"  Average MLE Epoch Loss: {average_epoch_loss:.4f}")
            logger.info(f"  Optimizer steps in MLE epoch: {optimizer_steps_in_epoch}")
            if last_clip_value_used is not None:
                logger.info(
                    f"  Last AutoClip threshold used: {last_clip_value_used:.4f}"
                )

            self.lr_scheduler.step()
            logger.info(
                f"  LR scheduler stepped. New LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if (epoch + 1) % save_every_epochs == 0 or (epoch + 1) == num_epochs:
                autoclipper_state = self.auto_clipper.grad_history \
                                    if hasattr(self, 'auto_clipper') else None
                self.save_checkpoint(
                    epoch=epoch + 1,
                    loss_history=self.epoch_losses,
                    autoclipper_state=autoclipper_state
                )
            logger.info("=" * 60)

        logger.info("Maximum Likelihood training finished.")
        self._plot_and_save_loss(plot_filename=f"mle_{self.LOSS_PLOT_FILENAME}")


    # --- Sampling/Solving Logic ---
    def _setup_solver(self, do_log: bool) -> ODESolver:
        """Configures and returns the ODESolver for sampling.

        This method sets up the ODESolver, potentially transforming the model's
        velocity field if a `sampling_path_scheduler` is different from the
        `training_path_scheduler`.

        Args:
            do_log: Whether to log information about the solver setup.

        Returns:
            An initialized ODESolver instance ready for sampling.
        """
        self.model.eval() # Ensure model is in evaluation mode for sampling
        velocity_model_for_solver = self.model_wrapper(self.model)

        if self.sampling_path_scheduler is not None and \
           self.sampling_path_scheduler != self.training_path_scheduler:
            # If a different scheduler is specified for sampling,
            # the model's velocity field needs to be transformed accordingly.
            if do_log:
                logger.info(
                    "Using ScheduleTransformedModel for sampling due to "
                    "different sampling path scheduler."
                )
            transformed_model = ScheduleTransformedModel(
                velocity_model=velocity_model_for_solver,
                original_scheduler=self.training_path_scheduler,
                new_scheduler=self.sampling_path_scheduler,
            )
            solver = ODESolver(velocity_model=transformed_model)
        else:
            # Use the original (wrapped) model's velocity field
            if do_log:
                logger.info(
                    "Using the original model's velocity field (via wrapper) "
                    "for sampling."
                )
            solver = ODESolver(velocity_model=velocity_model_for_solver)
        return solver

    def solve(
        self,
        solve_loader: DataLoader,
        num_samples: int,
        num_steps: int = DEFAULT_NUM_STEPS,
        method: str = DEFAULT_METHOD,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        time_grid: Tensor = DEFAULT_TIME_GRID_FWD,
        timestep_schedule_mu: float = 0.0,
        timestep_schedule_sigma: float = 1.0,
        for_gpu_warmup: bool = False,
        do_log: bool = True,
        reshape: bool = True,
    ) -> Tensor:
        """Generates samples by solving the ODE from $p_0$ to $p_1$.

        This method takes initial samples $x_0$ (typically from the base
        distribution, e.g., noise) and evolves them according to the learned
        ODE $dx/dt = v(x, t)$ to produce samples $x_1$ from the target
        distribution.

        Args:
            solve_loader: DataLoader providing initial samples ($x_0$) and
                optionally conditions.
            num_samples: Expected number of generated samples per original
                concept/label/initial noise if the `solve_loader` was
                constructed with repetitions. Used for reshaping the output if
                `reshape` is True.
            num_steps: Number of steps for fixed-step ODE solvers.
            method: ODE solver integration method (e.g., 'heun3', 'dopri5').
            atol: Absolute tolerance for adaptive ODE solvers.
            rtol: Relative tolerance for adaptive ODE solvers.
            time_grid: Time grid for the forward ODE solve (e.g., from 0.0 to 1.0).
            timestep_schedule_mu: Factor for scheduling the time grid.
            timestep_schedule_sigma: Factor for scheduling the time grid.
            for_gpu_warmup: If True, performs a GPU warmup without actual sampling.
            do_log: Whether to log the solving process.
            reshape: If True, attempts to reshape the output tensor of
                generated samples based on `num_samples`.

        Returns:
            A Tensor containing the generated samples. The shape might be
            `(num_original_items, num_samples, *sample_dims)` if `reshape`
            is True and successful, otherwise
            `(total_generated_samples, *sample_dims)`.
            Returns an empty tensor if `solve_loader` is empty.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Free up GPU memory before intensive sampling

        self.model.eval() # Ensure model is in evaluation mode
        solver = self._setup_solver(do_log=do_log)
        all_solutions_list: List[Tensor] = []

        if for_gpu_warmup:
            logger.info("Performing GPU warmup. No actual sampling will be done.")

        # Determine original and scheduled time grid for ODE solving
        if do_log:
            logger.info("Starting sample generation (solving ODE)...")
            logger.info(f"  Solver: {method}, Time Grid (Original): {time_grid.tolist()}")

        if method in ['euler', 'midpoint', 'rk4', 'heun2', 'heun3']: # Fixed-step
            time_grid_ode = torch.linspace(
                time_grid[0].item(), time_grid[-1].item(),
                steps=num_steps + 1, device=self.device
            )
            if do_log:
                logger.info(f"  Time Grid (Fixed Steps): {time_grid_ode.tolist()}")
        else: # Adaptive
            time_grid_ode = time_grid.to(self.device)
            if do_log:
                logger.info(f"  Tolerances for adaptive solver: atol={atol}, rtol={rtol}")

        # Apply timestep scheduling for sampling (e.g., variance exploding schedule)
        # 'a=0.3' is a hyperparameter for this specific scheduler, may differ from likelihood's 'a'
        time_grid_scheduled_for_ode = timestep_scheduler(
            time_grid_ode,
            mu=timestep_schedule_mu,
            sigma=timestep_schedule_sigma,
        )
        if do_log:
            logger.info(f"  Time Grid (Scheduled for ODE): {time_grid_scheduled_for_ode.tolist()}")


        with torch.no_grad(): # Sampling is an evaluation pass
            progress_bar = tqdm(
                solve_loader, desc="Solving ODE (Sampling)",
                leave=False, total=len(solve_loader)
            )
            for batch_content in progress_bar:
                current_model_extras = dict()
                if isinstance(batch_content, (list, tuple)):
                    x_init_batch = batch_content[0].to(self.device)
                    if len(batch_content) > 1:
                        if isinstance(batch_content[1], Tensor):
                            conds = batch_content[1].to(self.device)
                            current_model_extras['conditions'] = [conds,]
                        elif isinstance(batch_content[1], list):
                            conds = [c.to(self.device) for c in batch_content[1]]
                            current_model_extras['conditions'] = conds
                        else:
                            raise TypeError(
                                "Label must be a Tensor or a list of Tensors. "
                                f"Received type: {type(batch_content[1])}"
                            )
                else:
                    x_init_batch = batch_content.to(self.device)
                    current_model_extras.pop('conditions', None) # Clear if not provided

                # ODESolver.sample returns the solution at the final time point
                solution_batch = solver.sample(
                    x_init=x_init_batch,
                    step_size=None, # Handled by num_steps or adaptive solver
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    time_grid=time_grid_scheduled_for_ode,
                    return_intermediates=False, # Only need final samples
                    enable_grad=False,
                    **current_model_extras
                )
                # Move to CPU before appending to save GPU memory, esp. if solutions are large
                all_solutions_list.append(solution_batch.cpu())

                if for_gpu_warmup:
                    break

        if for_gpu_warmup:
            logger.info("GPU warmup completed. Returning empty tensor.")
            # Return an empty tensor for GPU warmup, shape can be adjusted if needed
            return torch.empty((0,), dtype=torch.float32, device='cpu')

        if not all_solutions_list:
             logger.warning(
                 "No solutions were generated (solve_loader might be empty). "
                 "Returning an empty tensor."
             )
             # Determine dtype from model parameters or default to float32
             # The shape of the empty tensor should ideally match an expected sample shape,
             # but without a sample, a 1D empty tensor is a safe fallback.
             example_param = next(self.model.parameters(), None)
             dtype = example_param.dtype if example_param is not None else torch.float32
             # To be more robust, if solve_loader had an item, its shape could be used.
             # For now, return 0 items with minimal dimensionality.
             return torch.empty((0,), dtype=dtype, device='cpu')

        concatenated_solutions = torch.cat(all_solutions_list, dim=0)
        final_solutions_tensor = concatenated_solutions # Default if reshape fails or is false

        # --- Reshape Logic (if requested) ---
        if reshape:
            try:
                dataset_size = len(solve_loader.dataset) \
                               if hasattr(solve_loader, 'dataset') and \
                                  solve_loader.dataset is not None else None
                actual_items_processed_total = concatenated_solutions.shape[0]
                sample_dims = concatenated_solutions.shape[2:] # Dimensions of a single sample

                if num_samples <= 0:
                     logger.warning(
                         f"num_samples ({num_samples}) is not positive. "
                         "Cannot perform meaningful reshape. Returning concatenated solutions."
                     )
                     num_original_items = actual_items_processed_total
                     effective_num_samples_per_item = 1
                elif dataset_size is not None:
                    if dataset_size % num_samples != 0:
                        logger.warning(
                            f"Dataset size ({dataset_size}) is not perfectly "
                            f"divisible by num_samples ({num_samples}). Reshape might "
                            "group items unexpectedly."
                        )
                    num_original_items = dataset_size // num_samples
                    effective_num_samples_per_item = num_samples
                    if actual_items_processed_total != dataset_size:
                        logger.warning(
                            f"Total generated samples ({actual_items_processed_total}) "
                            f"does not match solve_loader dataset size ({dataset_size}). "
                            "Reshape will be based on actual generated samples."
                        )
                        num_original_items = actual_items_processed_total // num_samples
                else: # Dataset size unknown
                    if actual_items_processed_total % num_samples != 0:
                        logger.warning(
                            f"Total generated samples ({actual_items_processed_total}) "
                            f"is not perfectly divisible by num_samples ({num_samples}). "
                            "Reshape might be inexact."
                        )
                    num_original_items = actual_items_processed_total // num_samples
                    effective_num_samples_per_item = num_samples


                if num_original_items > 0 and \
                   actual_items_processed_total == num_original_items * effective_num_samples_per_item:
                    # Target shape: [num_original_items, num_samples_per_item, *sample_dims]
                    final_solutions_tensor = concatenated_solutions.reshape(
                        num_original_items, effective_num_samples_per_item, *sample_dims
                    )
                    if do_log:
                        logger.info(
                            f"Reshaped generated solutions tensor shape: {final_solutions_tensor.shape}"
                        )
                else:
                    logger.warning(
                        f"Cannot reshape solutions as expected (num_original_items={num_original_items}, "
                        f"num_samples_per_item={effective_num_samples_per_item}, "
                        f"actual_generated={actual_items_processed_total}). "
                        f"Returning concatenated solutions with shape: {concatenated_solutions.shape}"
                    )
                    # final_solutions_tensor remains concatenated_solutions

            except Exception as e: # pragma: no cover
                logger.error(
                    f"Error during solution reshaping. Concatenated shape: "
                    f"{concatenated_solutions.shape}. Error: {e}. "
                    "Returning concatenated solutions."
                )
                # final_solutions_tensor remains concatenated_solutions
        else: # reshape is False
            if do_log:
                logger.info(
                    f"Generated solutions concatenated shape (reshape=False): "
                    f"{final_solutions_tensor.shape}"
                )
        # --- End Reshape Logic ---

        if do_log:
            logger.info(
                f"Sample generation finished. Total samples: "
                f"{final_solutions_tensor.shape[0] if not reshape else final_solutions_tensor.shape[0] * final_solutions_tensor.shape[1]}.")


        return final_solutions_tensor

    # --- Checkpointing ---
    def save_checkpoint(
        self,
        epoch: int,
        loss_history: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """Saves model, optimizer, scheduler, and training metadata.

        Args:
            epoch: The current completed epoch number (1-based).
            loss_history: List of average losses per epoch up to this point.
            **kwargs: Other optional data to include in the checkpoint,
                e.g., `autoclipper_state`.
        """
        checkpoint_data = {
            'epoch': epoch, # Last completed epoch
            'model_params': self.model_params, # Original model hyperparameters
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss_history': loss_history if loss_history is not None else [],
            'autoclipper_state': kwargs.get('autoclipper_state', None)
        }

        checkpoint_dir = self._setup_save_subdir(dir_name=self.CHECKPOINT_DIR_NAME)
        checkpoint_filename = f'checkpoint_epoch_{epoch:04d}.pth'
        checkpoint_save_path = checkpoint_dir / checkpoint_filename

        try:
            torch.save(checkpoint_data, checkpoint_save_path)
            logger.info(
                f"Checkpoint saved successfully to {checkpoint_save_path} "
                f"(Epoch {epoch})"
            )
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to save checkpoint to {checkpoint_save_path}: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads state from a checkpoint file to resume training or for inference.

        This method loads the model weights, optimizer state, learning rate
        scheduler state, completed epoch number, loss history, and AutoClip
        state (if available) from the specified checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pth).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            IsADirectoryError: If the checkpoint path is a directory.
            ValueError: If the checkpoint file is corrupt or missing critical keys.
            RuntimeError: If there's an error during state loading (e.g.,
                architecture mismatch).
        """
        load_path = Path(checkpoint_path)
        if not load_path.exists():
            logger.error(f"Checkpoint file not found: {load_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")
        if not load_path.is_file():
            logger.error(f"Checkpoint path is not a file: {load_path}")
            raise IsADirectoryError(f"Checkpoint path is not a file: {load_path}")

        try:
            # Load checkpoint onto the device specified during CFMExecutor initialization
            # weights_only=False allows loading optimizer, scheduler, etc.
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

            # --- Load core states ---
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            # --- Load training progress ---
            # 'epoch' in checkpoint is the last completed epoch.
            # starting_epoch will be this value, so training resumes at epoch + 1.
            self.starting_epoch = checkpoint.get('epoch', 0)
            self.epoch_losses = checkpoint.get('loss_history', [])

            # --- Load AutoClip state if available ---
            autoclipper_state = checkpoint.get('autoclipper_state', None)
            if autoclipper_state is not None and hasattr(self, 'auto_clipper'):
                self.auto_clipper.grad_history = autoclipper_state
                logger.info("Loaded AutoClip gradient history from checkpoint.")
            elif hasattr(self, 'auto_clipper'):
                logger.info(
                    "No AutoClip state found in checkpoint. AutoClip will "
                    "start with an empty gradient history."
                )

            # --- Log loaded information ---
            logger.info(
                f"Loaded checkpoint from {load_path}. "
                f"Last completed epoch: {self.starting_epoch}."
            )
            logger.info(f"Training will resume from epoch {self.starting_epoch + 1}.")

            if self.epoch_losses:
                try:
                    last_recorded_loss = self.epoch_losses[-1]
                    logger.info(f"  Last recorded average epoch loss: {last_recorded_loss:.4f}")
                except IndexError: # pragma: no cover
                     logger.info("  Loss history found in checkpoint but is empty.")
            else:
                logger.info("  No epoch loss history found in checkpoint or history is empty.")

            if 'model_params' in checkpoint and checkpoint['model_params']:
                logger.info(f"  Associated model params from checkpoint: {checkpoint['model_params']}")
            else:
                logger.info("  No model parameters (model_params) found in checkpoint.")

        except FileNotFoundError: # Should be caught by pre-check, but for safety
            logger.error(f"Checkpoint file not found during actual load: {load_path}")
            raise
        except KeyError as e:
            logger.error(f"Checkpoint file {load_path} is missing an expected key: {e}")
            # Distinguish critical missing keys
            critical_keys = ['model_state_dict', 'optimizer_state_dict', 'lr_scheduler_state_dict']
            if str(e).strip("'") in critical_keys:
                 logger.error(
                     "Critical state dictionary missing. Checkpoint may be corrupt "
                     "or from an incompatible version."
                 )
            raise ValueError(
                f"Checkpoint file {load_path} is corrupt or has an unexpected "
                f"format (missing key: {e})."
            ) from e
        except Exception as e: # Catch other loading errors (e.g., architecture mismatch)
            logger.error(f"An error occurred while loading checkpoint {load_path}: {e}", exc_info=True)
            logger.error(
                "This can be due to model architecture mismatches, optimizer type "
                "changes, or other incompatibilities between the saved state and "
                "the current setup."
            )
            raise RuntimeError(f"Failed to load states from checkpoint {load_path}.") from e
