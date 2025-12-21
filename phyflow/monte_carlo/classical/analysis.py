import torch
from torch import Tensor
from typing import Tuple, Dict, Callable, Any, List


class JackknifeAnalysis:
    """Performs Jackknife error analysis on Monte Carlo data for the XY Model.

    This class processes raw Monte Carlo samples to compute physical observables
    and their associated statistical errors using the Jackknife resampling method.

    Key Features:
        1. Reduces (B, N, L, L) raw samples to (B, N) time-series observables.
        2. Implements binning to mitigate autocorrelation effects.
        3. uses Jackknife resampling to correctly estimate errors for non-linear
           observables (e.g., Specific Heat, Susceptibility, Stiffness).

    Attributes:
        model: The XY model instance containing system parameters.
        samples: The raw simulation samples tensor.
        device: The computing device (CPU or CUDA).
        B: Batch size (number of parallel chains/temperatures).
        N: Number of samples per chain.
        L: Linear lattice size.
        T: Temperature tensor.
        J: Coupling constant.
        raw_obs: Dictionary containing pre-computed raw observables.
    """

    def __init__(self, model: Any, samples: Tensor):
        """Initializes the JackknifeAnalysis class.

        Args:
            model: An instance of the XYModel containing attributes `J`, `T`,
                and `L`.
            samples: The output tensor from the model's forward pass, with
                shape [Batch, Total_Samples, L, L].
        """
        self.model = model
        self.samples = samples
        self.device = samples.device
        self.B, self.N, self.L, _ = samples.shape
        self.T = model.T  # Shape: [B]
        self.J = model.J

        # Pre-compute raw time series observables.
        # This step reduces spatial dimensions (L, L) to optimize memory usage.
        print("Pre-computing raw observables from samples...")
        self.raw_obs = self._compute_raw_observables()
        print("Raw observables computed.")

    def _compute_raw_observables(self) -> Dict[str, Tensor]:
        """Computes fundamental physical observables for each sample.

        Returns:
            A dictionary containing tensors of shape [B, N]:
            - 'E': Total energy per site.
            - 'M': Magnetization density.
            - 'Upsilon_y': Sum of cosine differences along the Y-axis (for stiffness).
            - 'I_y': Sum of sine differences along the Y-axis (for stiffness).
        """
        # Flatten Batch and Sample dimensions to prevent Out-Of-Memory (OOM) errors.
        # flat_samples shape: [B*N, L, L]
        flat_samples = self.samples.view(-1, self.L, self.L)

        obs = {}

        # Use no_grad as this is purely post-processing analysis.
        with torch.no_grad():
            # 1. Prepare neighbor data
            # theta shape: [B*N, L, L]
            theta = flat_samples

            # For the XY model, we calculate differences between neighbors.
            # We focus on the Y-direction (dim=-2) for Stiffness calculations.
            # t_up corresponds to (i+1, j) -> Y-direction neighbor
            t_up = torch.roll(theta, shifts=-1, dims=-2)
            # t_right corresponds to (i, j+1) -> X-direction neighbor
            t_right = torch.roll(theta, shifts=-1, dims=-1)

            diff_y = t_up - theta
            diff_x = t_right - theta

            # --- Energy (per site) ---
            # Hamiltonian: H = -J * sum(cos(theta_i - theta_j))
            e_local_sum = torch.cos(diff_y).sum(dim=(1, 2)) + \
                          torch.cos(diff_x).sum(dim=(1, 2))
            obs['E'] = -self.J * e_local_sum / (self.L ** 2)

            # --- Magnetization (per site) ---
            # M = |sum(exp(i*theta))| / L^2
            mx = torch.cos(theta).sum(dim=(1, 2))
            my = torch.sin(theta).sum(dim=(1, 2))
            obs['M'] = torch.sqrt(mx ** 2 + my ** 2) / (self.L ** 2)

            # --- Stiffness Components (unnormalized sums) ---
            # Stiffness Formula: rho_s = (1/L^2) * [ <Upsilon> - (1/T)*<I^2> ]
            # Upsilon_y = J * sum(cos(theta_i+1 - theta_i))  (Link Term)
            # I_y = J * sum(sin(theta_i+1 - theta_i))        (Current Term)
            # Note: We store the raw sums here. Normalization by L^2 and T
            # occurs during the Jackknife averaging.
            obs['Upsilon_y'] = self.J * torch.cos(diff_y).sum(dim=(1, 2))
            obs['I_y'] = self.J * torch.sin(diff_y).sum(dim=(1, 2))

        # Reshape all observables back to [B, N]
        for k, v in obs.items():
            obs[k] = v.view(self.B, self.N)

        return obs

    def _perform_binning(self, data: Tensor, bin_size: int) -> Tensor:
        """Bins the data to reduce autocorrelation.

        Args:
            data: Input tensor of shape [B, N].
            bin_size: Number of raw samples to aggregate into one bin.

        Returns:
            A tensor of shape [B, K], where K = N // bin_size.

        Raises:
            ValueError: If the resulting number of bins is less than 2.
        """
        B, N = data.shape
        K = N // bin_size
        if K < 2:
            raise ValueError(
                f"Bin size {bin_size} is too large for {N} samples. "
                "At least 2 bins are required for error analysis."
            )

        # Discard trailing data that doesn't fit into a full bin
        trimmed_data = data[:, :K * bin_size]

        # Reshape to [B, K, bin_size] and compute mean across the bin dimension
        return trimmed_data.view(B, K, bin_size).mean(dim=2)

    def _compute_jackknife_estimate(
            self,
            inputs: Tuple[Tensor, ...],
            func: Callable[[List[Tensor], Tensor], Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Computes the Jackknife mean and error for a general function.

        Args:
            inputs: A tuple of binned tensors (e.g., (E_binned, E2_binned)),
                each with shape [B, K].
            func: A function that accepts a list of jackknife means (subset averages)
                and the temperature tensor, returning the physical observable.

        Returns:
            A tuple (mean, error), both tensors of shape [B].
        """
        n_bins = inputs[0].shape[1]

        # 1. Compute Global Means over all bins
        # List of [B] tensors
        global_means = [x.mean(dim=1) for x in inputs]

        # 2. Construct Jackknife Samples (Leave-one-out means)
        # Sums: [B] -> expanded to [B, 1] for broadcasting
        sums = [x.sum(dim=1, keepdim=True) for x in inputs]

        # Jackknife means: x_J[i] = (Sum - x[i]) / (K - 1)
        # Result is list of [B, K] tensors
        jack_means_list = [(s - x) / (n_bins - 1) for s, x in zip(sums, inputs)]

        # 3. Compute observable for each Jackknife sample
        # Expand Temperature to [B, 1] to match jackknife samples
        t_expanded = self.T.view(-1, 1)
        jack_estimates = func(jack_means_list, t_expanded)  # Returns [B, K]

        # 4. Compute Jackknife Mean and Error
        # The bias-corrected estimator mean
        bar_f = jack_estimates.mean(dim=1)  # [B]

        # Jackknife Variance Formula: ((K-1)/K) * sum((f_i - bar_f)^2)
        variance_jack = ((jack_estimates - bar_f.unsqueeze(1)) ** 2).sum(dim=1)
        error = torch.sqrt((n_bins - 1) / n_bins * variance_jack)

        return bar_f, error

    def compute_all_errors(self, bin_size: int = 100) -> Dict[str, Dict[str, Tensor]]:
        """Computes means and Jackknife errors for all major observables.

        Args:
            bin_size: The number of raw samples per bin. Should be chosen such
                that it is larger than the autocorrelation time (tau).

        Returns:
            A dictionary where keys are observable names ('Energy', 'Specific_Heat',
            etc.) and values are dictionaries containing:
                - 'mean': Tensor of shape [B]
                - 'error': Tensor of shape [B]
        """
        results = {}

        # 1. Prepare Binned Data
        # We bin the raw observables and their squares to compute variances later.

        # Energy terms
        e_binned = self._perform_binning(self.raw_obs['E'], bin_size)  # <E>
        e2_binned = self._perform_binning(self.raw_obs['E'] ** 2, bin_size)  # <E^2>

        # Magnetization terms
        m_binned = self._perform_binning(self.raw_obs['M'], bin_size)  # <M>
        m2_binned = self._perform_binning(self.raw_obs['M'] ** 2, bin_size)  # <M^2>

        # Stiffness terms
        up_binned = self._perform_binning(self.raw_obs['Upsilon_y'], bin_size)  # <Upsilon>
        i2_binned = self._perform_binning(self.raw_obs['I_y'] ** 2, bin_size)  # <I^2>

        # --- Define Observable Functions (accepting Jackknife means) ---

        # 1. Energy: E = <E>
        def func_energy(args: List[Tensor], t: Tensor) -> Tensor:
            return args[0]

        results['Energy'] = {}
        results['Energy']['mean'], results['Energy']['error'] = \
            self._compute_jackknife_estimate((e_binned,), func_energy)

        # 2. Magnetization: M = <M>
        def func_mag(args: List[Tensor], t: Tensor) -> Tensor:
            return args[0]

        results['Magnetization'] = {}
        results['Magnetization']['mean'], results['Magnetization']['error'] = \
            self._compute_jackknife_estimate((m_binned,), func_mag)

        # 3. Specific Heat: Cv
        # Formula: Cv = (L^2 / T^2) * (<E^2> - <E>^2)
        # Note: raw_obs['E'] is energy per site.
        # Var(E_total) = L^4 * Var(E_site).
        # Cv = Var(E_total) / T^2 / L^2 = L^2 * Var(E_site) / T^2.
        def func_cv(args: List[Tensor], t: Tensor) -> Tensor:
            e_mean, e2_mean = args
            var_e = e2_mean - e_mean ** 2
            return (self.L ** 2 / t ** 2) * var_e

        results['Specific_Heat'] = {}
        results['Specific_Heat']['mean'], results['Specific_Heat']['error'] = \
            self._compute_jackknife_estimate((e_binned, e2_binned), func_cv)

        # 4. Susceptibility: Chi
        # Formula: Chi = (L^2 / T) * (<M^2> - <M>^2)
        # Note: raw_obs['M'] is magnetization per site.
        def func_chi(args: List[Tensor], t: Tensor) -> Tensor:
            m_mean, m2_mean = args
            var_m = m2_mean - m_mean ** 2
            return (self.L ** 2 / t) * var_m

        results['Susceptibility'] = {}
        results['Susceptibility']['mean'], results['Susceptibility']['error'] = \
            self._compute_jackknife_estimate((m_binned, m2_binned), func_chi)

        # 5. Spin Stiffness: rho_s
        # Formula: (1/L^2) * [ <Upsilon> - (1/T)*<I^2> ]
        # Inputs are sum-over-lattice quantities (unnormalized).
        def func_stiffness(args: List[Tensor], t: Tensor) -> Tensor:
            up_mean, i2_mean = args
            return (up_mean - i2_mean / t) / (self.L ** 2)

        results['Stiffness'] = {}
        results['Stiffness']['mean'], results['Stiffness']['error'] = \
            self._compute_jackknife_estimate((up_binned, i2_binned), func_stiffness)

        return results

    def print_report(self, results: Dict[str, Dict[str, Tensor]]) -> None:
        """Prints a formatted table of the analysis results."""
        header = (
            f"{'Temp':<10} | {'Energy':<20} | {'Cv':<20} | "
            f"{'Magnetization':<20} | {'Chi':<20} | {'Stiffness':<20}"
        )
        print(header)
        print("-" * 120)

        # Convert temperature to CPU numpy for iteration
        t_cpu = self.T.cpu().numpy()

        def get_fmt_str(name: str, idx: int) -> str:
            val = results[name]['mean'][idx].item()
            err = results[name]['error'][idx].item()
            return f"{val:.5f} ± {err:.5f}"

        for i, t_val in enumerate(t_cpu):
            row = (
                f"{t_val:<10.4f} | "
                f"{get_fmt_str('Energy', i):<20} | "
                f"{get_fmt_str('Specific_Heat', i):<20} | "
                f"{get_fmt_str('Magnetization', i):<20} | "
                f"{get_fmt_str('Susceptibility', i):<20} | "
                f"{get_fmt_str('Stiffness', i):<20}"
            )
            print(row)