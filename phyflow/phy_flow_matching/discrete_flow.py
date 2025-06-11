# -*- coding: utf-8 -*-
"""
This module implements core components for training and utilizing
Discrete Flow Matching (DFM) models, including data handling utilities,
adaptive gradient clipping, and an executor class for managing the
training, and sampling processes for discrete data.
"""

# --- Standard Imports ---
import random
import time
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Iterator, Any, Union, Callable

import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.nn import functional as F

# --- Imports from flow_matching package ---
from flow_matching.path import (
    MixtureDiscreteProbPath,
    DiscretePathSample
)
from flow_matching.path.scheduler import (
    ConvexScheduler,
    PolynomialConvexScheduler # A common choice for DFM
)
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.utils import ModelWrapper # Assuming ModelWrapper is general enough

# Attempt to import matplotlib for plotting, but make it optional
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

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
        if not iterators:
            break

        idx = random.randrange(len(iterators))
        try:
            batch = next(iterators[idx])
        except StopIteration:
            logger.debug(f"DataLoader at index {idx} exhausted. Resetting.")
            iterators[idx] = iter(dataloaders[idx])
            try:
                batch = next(iterators[idx])
            except StopIteration:
                logger.warning(
                    f"DataLoader at index {idx} seems empty even after reset. "
                    "Skipping this selection."
                )
                continue
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
            return 0.0

        for p in params_with_grad:
            grad_norm = p.grad.detach().norm(2).item()
            total_norm_sq += grad_norm ** 2
        return total_norm_sq ** 0.5

    def __call__(self) -> Optional[float]:
        """Computes gradient norm, updates history, and applies clipping.

        Returns:
            The clipping threshold value used. Returns `None` if no clipping
            was performed.
        """
        current_norm = self._compute_grad_norm()

        if current_norm > 1e-6:
            self.grad_history.append(current_norm)

        if self.window_size is not None and \
           len(self.grad_history) > self.window_size:
            self.grad_history = self.grad_history[-self.window_size:]

        if not self.grad_history:
            logger.debug("AutoClip: Gradient history is empty. Skipping clipping.")
            return None

        clip_val = float(np.percentile(self.grad_history, self.percentile))

        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
            return clip_val
        else:
            logger.debug(
                f"AutoClip: Calculated clip value ({clip_val:.4f}) is not "
                "positive. Skipping clipping."
            )
            return None


class DFMExecutor:
    """Manages training, evaluation, and sampling for Discrete Flow Matching.

    This executor class encapsulates common workflows for Discrete Flow Matching
    (DFM) models, particularly for sequence generation or other discrete data
    modalities. It handles model training loops and sample generation using
    a discrete solver.

    Default Constants:
        DEFAULT_SAVE_EVERY_EPOCHS: Frequency for saving checkpoints (epochs).
        DEFAULT_SAMPLING_STEPS: Default number of steps for discrete solver.
        CHECKPOINT_DIR_NAME: Subdirectory name for checkpoints.
        LOSS_PLOT_FILENAME: Filename for the training loss plot.
        DEFAULT_GRAD_ACCUM_STEPS: Default gradient accumulation steps.
        DEFAULT_AUTOCLIP_PERCENTILE: Default percentile for AutoClip.
        DEFAULT_AUTOCLIP_WINDOW_SIZE: Default window size for AutoClip.
        DEFAULT_SAMPLING_TIME_GRID: Default time grid for sampling (0 to ~1).
                                   Slightly less than 1 to avoid potential
                                   singularities with some schedulers if kappa_t=1.
    """

    # --- Default constants ---
    DEFAULT_SAVE_EVERY_EPOCHS: int = 1
    DEFAULT_SAMPLING_STEPS: int = 100 # Common for discrete processes
    CHECKPOINT_DIR_NAME: str = 'checkpoints_dfm'
    LOSS_PLOT_FILENAME: str = 'training_loss_dfm_plot.png'
    DEFAULT_GRAD_ACCUM_STEPS: int = 1
    DEFAULT_AUTOCLIP_PERCENTILE: float = 10.0
    DEFAULT_AUTOCLIP_WINDOW_SIZE: int = 1000
    # Time grid typically from 0 to a value slightly less than 1 for mixture paths
    # to avoid kappa_t = 1 where 1-kappa_t becomes 0 in velocity calcs.
    DEFAULT_SAMPLING_TIME_GRID: Tensor = torch.tensor([0.0, 1.0 - 1e-3])


    def __init__(
        self,
        save_dir_root: str,
        model_params: dict, # Parameters of the model architecture
        model: nn.Module,   # The neural network model (e.g., a Transformer)
        optimizer: optim.Optimizer,
        lr_scheduler: LRScheduler,
        vocabulary_size: int, # Size of the discrete vocabulary
        training_path_scheduler: ConvexScheduler = PolynomialConvexScheduler(n=1.0),
        loss_fn_type: str = "cross_entropy", # "cross_entropy" or "generalized_kl"
        generalized_kl_path: Optional[MixtureDiscreteProbPath] = None, # Required if loss_fn_type is "generalized_kl"
        source_distribution_p_solver: Optional[Tensor] = None, # For MixtureDiscreteEulerSolver's div_free term
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        autoclip_percentile: float = DEFAULT_AUTOCLIP_PERCENTILE,
        autoclip_window_size: int = DEFAULT_AUTOCLIP_WINDOW_SIZE,
    ) -> None:
        """Initializes the DFMExecutor.

        Args:
            save_dir_root: Root directory for saving checkpoints, plots, etc.
            model_params: Dictionary of model hyperparameters.
            model: The neural network model.
            optimizer: The PyTorch optimizer.
            lr_scheduler: The PyTorch learning rate scheduler.
            vocabulary_size: The size of the discrete vocabulary.
            training_path_scheduler: Scheduler for defining the discrete
                probability path (kappa_t) during training.
            loss_fn_type: Type of loss function to use.
                "cross_entropy": Standard cross-entropy on model's output logits
                                 against the target state X1.
                "generalized_kl": Uses MixturePathGeneralizedKL. Requires
                                  `generalized_kl_path`.
            generalized_kl_path: The MixtureDiscreteProbPath instance to be used
                with MixturePathGeneralizedKL. Required if `loss_fn_type` is
                "generalized_kl".
            source_distribution_p_solver: Source distribution p for the
                MixtureDiscreteEulerSolver, used if div_free > 0 during sampling.
                Shape [vocabulary_size].
            device: The computation device ('cuda' or 'cpu').
            autoclip_percentile: Percentile for AutoClip.
            autoclip_window_size: Window size for AutoClip.

        Raises:
            OSError: If the `save_dir_root` cannot be created.
            NotADirectoryError: If `save_dir_root` is not a directory.
            ValueError: If AutoClip parameters are invalid or loss_fn_type is
                        "generalized_kl" but generalized_kl_path is not provided.
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
        self.optimizer: optim.Optimizer = optimizer
        self.lr_scheduler: LRScheduler = lr_scheduler
        self.vocabulary_size: int = vocabulary_size
        self.training_path_scheduler: ConvexScheduler = training_path_scheduler
        self.loss_fn_type: str = loss_fn_type
        self.source_distribution_p_solver = source_distribution_p_solver.to(self.device) \
            if source_distribution_p_solver is not None else None


        if self.loss_fn_type == "generalized_kl":
            if generalized_kl_path is None:
                raise ValueError(
                    "generalized_kl_path must be provided if loss_fn_type is "
                    "'generalized_kl'."
                )
            self.loss_fn: nn.Module = MixturePathGeneralizedKL(path=generalized_kl_path)
            logger.info("Using MixturePathGeneralizedKL loss.")
        elif self.loss_fn_type == "cross_entropy":
            # Actual loss calculation will be F.cross_entropy(model_output_logits.transpose(1,2), x1_target)
            # No specific nn.Module instance stored here, handled in _train_step
            self.loss_fn = None # Placeholder, logic is in _train_step
            logger.info("Using CrossEntropyLoss (via F.cross_entropy).")
        else:
            raise ValueError(f"Unsupported loss_fn_type: {loss_fn_type}")

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

        logger.info(f"DFMExecutor initialized. Device: {self.device}")
        if torch.cuda.is_available() and self.device == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                logger.info(f"GPU Name: {gpu_name} - {gpu_mem} MB")
            except Exception as e:
                logger.warning(f"Could not get GPU details: {e}")

        logger.info(f"Save directory root: {self.save_dir_root}")
        logger.info(f"Vocabulary size: {self.vocabulary_size}")

        if not _MATPLOTLIB_AVAILABLE:
            logger.warning(
                "Matplotlib not installed. Loss plotting will be disabled. "
                "Run 'pip install matplotlib' to enable."
            )
        if self.source_distribution_p_solver is not None:
             if self.source_distribution_p_solver.shape != torch.Size([vocabulary_size]):
                logger.warning(
                    f"source_distribution_p_solver shape {self.source_distribution_p_solver.shape} "
                    f"does not match vocabulary_size {vocabulary_size}. This might cause issues "
                    "if div_free > 0 during sampling."
                )


    def _setup_save_subdir(self, dir_name: str) -> Path:
        """Creates and returns a subdirectory within the root save directory."""
        target_dir = self.save_dir_root / dir_name
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating directory {target_dir}: {e}")
            raise
        return target_dir

    def set_save_dir_root(self, dir_path: str) -> None:
        """Sets a new root directory for saving results."""
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

    # --- Core Training Logic ---
    def _train_step(
        self,
        x_0_batch: Tensor, # Source discrete data (e.g., mask tokens or noisy data)
        x_1_batch: Tensor, # Target discrete data
        prob_path: MixtureDiscreteProbPath,
        gradient_accumulation_steps: int = 1
    ) -> float:
        """Performs a single training step for DFM.

        This involves:
        1. Sampling time `t`.
        2. Sampling `x_t` from the discrete probability path `p_t(X_t|X_0, X_1)`.
        3. Getting model's prediction (logits for posterior p(X_1|X_t)).
        4. Computing the loss (e.g., cross-entropy or generalized KL).
        5. Scaling loss for gradient accumulation and backpropagation.

        Args:
            x_0_batch: The source data batch (e.g., sequence of token IDs).
            x_1_batch: The target data batch.
            prob_path: The discrete probability path generator.
            gradient_accumulation_steps: Factor for scaling loss.

        Returns:
            The unscaled loss value for this step.
        """
        x_0, x_1 = x_0_batch.to(self.device), x_1_batch.to(self.device)

        # Sample time t ~ U[0, 1-eps] to avoid kappa_t = 1 issues.
        # See PDF section 7.5.4, Code 10 for t sampling.
        # Using a small epsilon.
        time_epsilon = 1e-3
        t_raw = torch.rand(x_1.shape[0], device=self.device) * (1.0 - time_epsilon)
        # Ensure t has the same number of dimensions as data if model expects that,
        # or keep as (batch_size,) if model handles broadcasting.
        # For many sequence models, t is (batch_size,).
        # If model needs t per token, it should expand it internally or path sampler needs it.
        # MixtureDiscreteProbPath expects t of shape (batch_size,)

        # Sample x_t from the mixture path P(X_t=X_1)=kappa_t, P(X_t=X_0)=1-kappa_t
        # No dx_t for discrete paths.
        path_sample: DiscretePathSample = prob_path.sample(t=t_raw, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t # Shape: (batch_size, seq_len) or similar

        # Get model's predicted logits for p(X_1|X_t).
        # Output shape: (batch_size, seq_len, vocab_size)
        # The model might take x_t and t (potentially expanded)
        # The PDF Code 11 suggests model(x_t, t) where t is (batch_size,)
        model_output_logits = self.model(x=x_t, time=t_raw) # Pass t_raw

        loss = 0.0
        if self.loss_fn_type == "cross_entropy":
            # Reshape for cross_entropy:
            # model_output_logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # x_1: (batch, seq_len) -> (batch * seq_len)
            loss = F.cross_entropy(
                model_output_logits.reshape(-1, self.vocabulary_size),
                x_1.reshape(-1)
            )
        elif self.loss_fn_type == "generalized_kl":
            # MixturePathGeneralizedKL expects logits.
            # It also needs x_1, x_t, and t.
            loss = self.loss_fn(
                logits=model_output_logits,
                x_1=x_1,
                x_t=x_t,
                t=t_raw # Pass the original t_raw used for path sampling
            )
        else:
            # Should not happen due to __init__ check
            raise RuntimeError(f"Internal error: Invalid loss_fn_type {self.loss_fn_type}")


        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        return loss.item()

    def train(
        self,
        train_loaders: Tuple[DataLoader, ...], # Each loader yields (x_0_batch, x_1_batch) or just x_1_batch
        num_epochs: int,
        gradient_accumulation_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
        save_every_epochs: int = DEFAULT_SAVE_EVERY_EPOCHS,
        fixed_x0_value: Optional[int] = None, # If x_0 is a fixed token ID (e.g. MASK)
        sample_x0_from_uniform: bool = False # If x_0 is sampled uniformly from vocab
    ) -> None:
        """Runs the main training loop for DFM.

        Args:
            train_loaders: Tuple of DataLoaders. Each loader should yield
                either (x_0_batch, x_1_batch) if source data is paired,
                or just x_1_batch if x_0 is to be generated (e.g., fixed or uniform).
            num_epochs: Total epochs for training.
            gradient_accumulation_steps: Accumulate gradients over this many batches.
            save_every_epochs: Frequency for saving checkpoints.
            fixed_x0_value: If not None, x_0 batches are created as tensors
                filled with this vocabulary index, matching x_1_batch shape.
            sample_x0_from_uniform: If True (and fixed_x0_value is None),
                x_0 batches are sampled uniformly from the vocabulary.
                If both are None, assumes DataLoaders provide (x_0, x_1) pairs.
        Raises:
            ValueError: If inputs are invalid.
            TypeError: If DataLoaders don't support len().
        """
        if not isinstance(gradient_accumulation_steps, int) or \
           gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        if not train_loaders:
            raise ValueError("train_loaders tuple cannot be empty.")
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(save_every_epochs, int) or save_every_epochs <= 0:
            raise ValueError("save_every_epochs must be a positive integer.")
        if fixed_x0_value is not None and sample_x0_from_uniform:
            raise ValueError("Cannot use both fixed_x0_value and sample_x0_from_uniform.")

        # Initialize the discrete probability path for training
        prob_path = MixtureDiscreteProbPath(scheduler=self.training_path_scheduler)
        self.model.train()

        try:
            total_batches_per_epoch = sum(len(dl) for dl in train_loaders)
            if total_batches_per_epoch == 0:
                raise ValueError("Total number of batches across all DataLoaders is zero.")
            logger.info(f"Total batches per epoch: {total_batches_per_epoch}")
        except TypeError:
            logger.error("One or more DataLoaders do not support len().")
            raise

        first_loader_batch_size = getattr(train_loaders[0], 'batch_size', None)
        if first_loader_batch_size:
            effective_batch_size = first_loader_batch_size * gradient_accumulation_steps
            logger.info(f"Effective batch size (approx): {effective_batch_size}")

        logger.info(f"Starting DFM training for {num_epochs} epochs.")
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

        self.optimizer.zero_grad()
        if self.starting_epoch == 0:
            self.epoch_losses = []

        for epoch in range(self.starting_epoch, num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            running_epoch_loss = 0.0
            processed_batches_in_epoch = 0
            processed_batches_in_accum_cycle = 0
            optimizer_steps_in_epoch = 0
            last_clip_value_used = None

            batch_iterator = multi_dataloader_random_cycle(
                train_loaders, total_batches_per_epoch
            )
            progress_bar = tqdm(
                batch_iterator,
                total=total_batches_per_epoch,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False
            )

            for batch_idx, batch_content in enumerate(progress_bar):
                x_0_batch, x_1_batch = None, None
                if isinstance(batch_content, (list, tuple)) and len(batch_content) == 2:
                    x_0_batch, x_1_batch = batch_content
                else: # Assumes batch_content is x_1_batch
                    x_1_batch = batch_content
                    if fixed_x0_value is not None:
                        x_0_batch = torch.full_like(x_1_batch, fixed_x0_value)
                    elif sample_x0_from_uniform:
                        x_0_batch = torch.randint(
                            0, self.vocabulary_size,
                            x_1_batch.shape,
                            dtype=x_1_batch.dtype
                         )
                    else:
                        raise ValueError(
                            "DataLoader yields single item, but neither "
                            "fixed_x0_value nor sample_x0_from_uniform is set."
                        )

                step_loss = self._train_step(
                    x_0_batch, x_1_batch, prob_path, gradient_accumulation_steps
                )
                running_epoch_loss += step_loss
                processed_batches_in_epoch += 1
                processed_batches_in_accum_cycle += 1

                is_last_batch_in_epoch = (batch_idx + 1) == total_batches_per_epoch
                if (processed_batches_in_accum_cycle == gradient_accumulation_steps or
                        is_last_batch_in_epoch):
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

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            average_epoch_loss = running_epoch_loss / total_batches_per_epoch \
                                 if total_batches_per_epoch > 0 else 0.0
            self.epoch_losses.append(average_epoch_loss)

            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f}s.")
            logger.info(f"  Average Epoch Loss: {average_epoch_loss:.4f}")
            logger.info(f"  Optimizer steps in epoch: {optimizer_steps_in_epoch}")
            if last_clip_value_used is not None:
                logger.info(f"  Last AutoClip threshold used: {last_clip_value_used:.4f}")

            self.lr_scheduler.step()
            logger.info(f"  LR scheduler stepped. New LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if (epoch + 1) % save_every_epochs == 0 or (epoch + 1) == num_epochs:
                autoclipper_state = self.auto_clipper.grad_history \
                                    if hasattr(self, 'auto_clipper') else None
                self.save_checkpoint(
                    epoch=epoch + 1,
                    loss_history=self.epoch_losses,
                    autoclipper_state=autoclipper_state
                )
            logger.info("-" * 60)

        logger.info("DFM Training finished.")
        self._plot_and_save_loss(plot_filename=self.LOSS_PLOT_FILENAME)

    def _plot_and_save_loss(self, plot_filename: str) -> None:
        """Plots the average training loss per epoch and saves the plot."""
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib is unavailable. Skipping loss plotting.")
            return
        if not self.epoch_losses:
            logger.info("No epoch loss data recorded; skipping plotting.")
            return

        try:
            plt.figure(figsize=(10, 6))
            epochs_to_plot = range(1, len(self.epoch_losses) + 1)
            plt.plot(
                epochs_to_plot, self.epoch_losses, marker='o',
                linestyle='-', label='Average Loss per Epoch'
            )
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.title('DFM Training Loss Curve')
            plt.legend()
            plt.grid(True)
            if len(list(epochs_to_plot)) < 20:
                plt.xticks(list(epochs_to_plot))
            else:
                plt.locator_params(axis='x', integer=True, nbins='auto')
            plt.tight_layout()
            save_path = self.save_dir_root / plot_filename
            plt.savefig(save_path)
            plt.close()
            logger.info(f"DFM training loss plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error plotting or saving DFM loss curve: {e}")


    def sample(
        self,
        x_init: Tensor, # Initial discrete data, e.g., (batch_size, seq_len)
        num_steps: int = DEFAULT_SAMPLING_STEPS,
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = DEFAULT_SAMPLING_TIME_GRID, # Typically [0, ~1]
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras # For conditional generation, e.g. class labels
    ) -> Tensor:
        """Generates discrete samples using the trained DFM model.

        This method uses the MixtureDiscreteEulerSolver to simulate the
        learned discrete flow process.

        Args:
            x_init: Initial discrete state tensor (e.g., batch of sequences
                    of token IDs). Shape: (batch_size, ...).
            num_steps: Number of steps for the discrete solver.
            div_free: Coefficient for the divergence-free term in the
                      probability velocity. See MixtureDiscreteEulerSolver docs.
            dtype_categorical: Precision for categorical sampler in solver.
            time_grid: Time grid for the solver. Typically from 0.0 to a value
                       slightly less than 1.0. If None, step_size must be used
                       implicitly by the solver based on num_steps over [0, time_grid_end].
                       The `MixtureDiscreteEulerSolver` expects a `step_size` OR a `time_grid`
                       passed to its sample method. Here we construct step_size
                       if time_grid implies it.
            return_intermediates: If True, return all intermediate states.
            verbose: Whether to show a progress bar.
            **model_extras: Additional keyword arguments passed to the
                            underlying model (e.g., conditions for conditional
                            generation).

        Returns:
            A Tensor containing the generated discrete samples.
            Shape: (num_time_steps, batch_size, ...) if return_intermediates,
                   else (batch_size, ...).
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.eval()

        # The MixtureDiscreteEulerSolver expects a ModelWrapper that outputs probabilities.
        # If self.model outputs logits, wrap it.
        class ProbabilityModel(ModelWrapper):
            def __init__(self, model: nn.Module):
                super().__init__(model)

            def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
                logits = self.model(x=x, t=t, **extras)
                return F.softmax(logits.to(dtype_categorical), dim=-1)

        probability_model_for_solver = ProbabilityModel(self.model)

        # Use the training path's scheduler for the solver path, or allow override
        solver_path = MixtureDiscreteProbPath(scheduler=self.training_path_scheduler)

        solver = MixtureDiscreteEulerSolver(
            model=probability_model_for_solver,
            path=solver_path, # DFM solver needs the path for kappa_t etc.
            vocabulary_size=self.vocabulary_size,
            source_distribution_p=self.source_distribution_p_solver
        )

        if verbose:
            logger.info("Starting DFM sample generation...")
            logger.info(f"  Num steps: {num_steps}, Time Grid (Target): {time_grid.tolist()}")


        # MixtureDiscreteEulerSolver takes step_size.
        # If time_grid is [t_start, t_end] and num_steps is given, calculate step_size.
        # Or, if time_grid has multiple points, it can be used directly if solver supports it.
        # The provided solver.sample expects step_size or uses time_grid for steps.
        # We'll use time_grid to define the interval and num_steps to define discretization.
        # Let's ensure time_grid is on the correct device.
        time_grid_solver = time_grid.to(x_init.device)

        # The solver's `sample` method can take a `time_grid` to define the
        # exact points for evaluation if `step_size` is None.
        # If we want `num_steps` over the interval defined by `time_grid`'s
        # min and max, we should prepare `t_discretization` for it.
        # However, the provided CFMExecutor's ODESolver directly takes `time_grid`
        # and `step_size` (where `step_size` can be None for adaptive or if
        # `time_grid` itself defines the steps).
        # `MixtureDiscreteEulerSolver.sample` uses `step_size` to make its own
        # `t_discretization` OR uses `time_grid` as `t_discretization` if `step_size` is None.

        # For clarity and consistency with typical discrete solver usage:
        # We use `num_steps` to define a step_size over the interval of `time_grid`.
        t_start_solve = time_grid_solver[0].item()
        t_end_solve = time_grid_solver[-1].item()
        calculated_step_size = (t_end_solve - t_start_solve) / num_steps

        # If return_intermediates is True, the solver's `time_grid` parameter
        # should be the points at which we want the output.
        # If `return_intermediates` is False, the solver's `time_grid`
        # effectively defines just the start and end for the `step_size` based integration.
        # The solver internally creates its own t_discretization if step_size is given.
        # Let's pass the full target time_grid if intermediates are needed,
        # otherwise just the boundary.
        solver_time_grid_param = time_grid_solver if return_intermediates else \
                                 torch.tensor([t_start_solve, t_end_solve], device=x_init.device)


        with torch.no_grad():
            solutions = solver.sample(
                x_init=x_init.to(self.device),
                step_size=calculated_step_size, # Pass step_size for num_steps behavior
                div_free=div_free,
                dtype_categorical=dtype_categorical,
                time_grid=solver_time_grid_param, # Defines interval and intermediate points if needed
                return_intermediates=return_intermediates,
                verbose=verbose,
                **model_extras
            )

        if verbose:
            logger.info("DFM sample generation finished.")
            final_shape = solutions.shape
            total_samples_str = f"{final_shape[1]} samples each of shape {final_shape[2:]}"
            if return_intermediates:
                total_samples_str = f"{final_shape[0]} timesteps for {total_samples_str}"

            logger.info(f"  Generated solutions shape: {final_shape} ({total_samples_str})")

        return solutions # Already on CPU if solver moved it, otherwise on device

    # --- Checkpointing ---
    def save_checkpoint(
        self,
        epoch: int,
        loss_history: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """Saves model, optimizer, scheduler, and training metadata for DFM.

        Args:
            epoch: The current completed epoch number (1-based).
            loss_history: List of average losses per epoch.
            **kwargs: Other optional data (e.g., `autoclipper_state`).
        """
        checkpoint_data = {
            'epoch': epoch,
            'model_params': self.model_params,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss_history': loss_history if loss_history is not None else [],
            'vocabulary_size': self.vocabulary_size, # Save vocab size
            'autoclipper_state': kwargs.get('autoclipper_state', None)
        }

        checkpoint_dir = self._setup_save_subdir(dir_name=self.CHECKPOINT_DIR_NAME)
        checkpoint_filename = f'dfm_checkpoint_epoch_{epoch:04d}.pth'
        checkpoint_save_path = checkpoint_dir / checkpoint_filename

        try:
            torch.save(checkpoint_data, checkpoint_save_path)
            logger.info(
                f"DFM Checkpoint saved successfully to {checkpoint_save_path} "
                f"(Epoch {epoch})"
            )
        except Exception as e:
            logger.error(f"Failed to save DFM checkpoint to {checkpoint_save_path}: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads state from a DFM checkpoint file.

        Args:
            checkpoint_path: Path to the DFM checkpoint file (.pth).

        Raises:
            FileNotFoundError, IsADirectoryError, ValueError, RuntimeError.
        """
        load_path = Path(checkpoint_path)
        if not load_path.exists():
            raise FileNotFoundError(f"DFM Checkpoint file not found: {load_path}")
        if not load_path.is_file():
            raise IsADirectoryError(f"DFM Checkpoint path is not a file: {load_path}")

        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

            loaded_vocab_size = checkpoint.get('vocabulary_size')
            if loaded_vocab_size is not None and loaded_vocab_size != self.vocabulary_size:
                logger.warning(
                    f"Vocabulary size in checkpoint ({loaded_vocab_size}) "
                    f"differs from executor's ({self.vocabulary_size}). "
                    "This might lead to issues."
                )
            # If vocab size wasn't saved, we assume current executor's is correct.

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self.starting_epoch = checkpoint.get('epoch', 0)
            self.epoch_losses = checkpoint.get('loss_history', [])

            autoclipper_state = checkpoint.get('autoclipper_state', None)
            if autoclipper_state is not None and hasattr(self, 'auto_clipper'):
                self.auto_clipper.grad_history = autoclipper_state
                logger.info("Loaded AutoClip gradient history from DFM checkpoint.")
            elif hasattr(self, 'auto_clipper'):
                logger.info("No AutoClip state found in DFM checkpoint.")

            logger.info(
                f"Loaded DFM checkpoint from {load_path}. "
                f"Last completed epoch: {self.starting_epoch}."
            )
            logger.info(f"DFM training will resume from epoch {self.starting_epoch + 1}.")
            # ... (rest of logging as in CFMExecutor)

        except KeyError as e:
            logger.error(f"DFM Checkpoint file {load_path} is missing key: {e}")
            raise ValueError(f"DFM Checkpoint {load_path} corrupt or unexpected format.") from e
        except Exception as e:
            logger.error(f"Error loading DFM checkpoint {load_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load states from DFM checkpoint {load_path}.") from e