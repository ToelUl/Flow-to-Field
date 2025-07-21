import torch
from torch import nn
import time
import gc
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Type, Union, Dict, Any

from .sampler import XYModel, IsingModel, PottsModel

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCDataGenerator(nn.Module):
    """
    Generates Monte Carlo simulation data for classical spin models.

    Initializes a specified sampler class, runs simulations across a
    temperature range, saves configurations and measurements, and generates plots.
    """

    def __init__(self,
                 sampler_class: Type[Union[XYModel, IsingModel, PottsModel]],
                 save_dir_root: Union[str, Path]):
        """
        Initializes the MCDataGenerator.

        Args:
            sampler_class: The uninitialized class of the sampler to use
                           (XYModel, IsingModel, or PottsModel).
            save_dir_root: The root directory where simulation data will be saved.
        """
        if not issubclass(sampler_class, (XYModel, IsingModel, PottsModel)):
            raise TypeError("sampler_class must be XYModel, IsingModel, or PottsModel")

        super().__init__()
        self.sampler_class = sampler_class
        self.sampler_name = sampler_class.__name__
        self.save_dir_root = Path(save_dir_root)
        logger.info(f"MCDataGenerator initialized for {self.sampler_name} with save root: {self.save_dir_root}")

    def _setup_save_subdir(self, dir_name: str) -> Path:
        """
        Creates a subdirectory within the root save directory if it doesn't exist.

        Args:
            dir_name: The name of the subdirectory to create relative to the
                      root save directory.

        Returns:
            The Path object representing the created or existing subdirectory.

        Raises:
            OSError: If directory creation fails for reasons other than it existing.
        """
        target_dir = self.save_dir_root / dir_name
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {target_dir}")
        except OSError as e:
            logger.error(f"Error creating directory {target_dir}: {e}")
            raise
        return target_dir

    def forward(self,
                 L: int,
                 T_start: float,
                 T_end: float,
                 precision: float,
                 ensemble_number: int,
                 device: str = "cuda:0",
                 q: int = None,
                 n_chains: int = 10,
                 pt_enabled: bool =None) -> None:
        """
        Runs the Monte Carlo simulation and data generation process.

        Args:
            L: Lattice size (linear dimension).
            T_start: Start temperature for the simulation range.
            T_end: End temperature for the simulation range.
            precision: Temperature step precision.
            ensemble_number: The number of samples (configurations) to generate
                             per temperature.
            device: The device to run the simulation on (e.g., "cuda:0" or "cpu").
            q: Number of states for Potts model (default is None, indicating not used).
            n_chains: Number of chains to run in parallel (default is 10).
            pt_enabled (bool, optional): Enable parallel tempering.
        """
        start_total_time = time.time()

        # --- 1. Determine Simulation Parameters ---
        T = torch.linspace(T_start, T_end, int((T_end - T_start) // precision) + 1, device=device)
        n_temps = len(T)

        if self.sampler_name == "XYModel":
            pt_enabled = True if pt_enabled is None else pt_enabled
            pt_interval = 1
            pt_prob = 0.1
            tau_pt = pt_interval / pt_prob
            tau = L**2.2 # Example autocorrelation time for XY model
            tau_eff = (tau_pt * tau) / (tau_pt + tau)
            factor_therm = 30
            factor_decorrelate = 2
            n_therm = int(factor_therm * tau)
            decorrelate = int(factor_decorrelate * tau_eff)
            n_sweeps = int(ensemble_number / n_chains) * decorrelate
            if decorrelate == 0:
                decorrelate = 1 # Avoid decorrelate=0
            if n_sweeps == 0:
                n_sweeps = decorrelate * int(ensemble_number / n_chains) if int(ensemble_number / n_chains)>0 else decorrelate # Ensure some sweeps if ensemble/chains is small

        elif self.sampler_name == "IsingModel":
            factor_therm = 5
            factor_decorrelate = 1
            tau = L**2
            n_therm = int(factor_therm * tau)
            decorrelate = int(factor_decorrelate * L) # Note: L not L**2
            n_sweeps = int(ensemble_number / n_chains) * decorrelate
            if decorrelate == 0:
                decorrelate = 1
            if n_sweeps == 0:
                n_sweeps = decorrelate * int(ensemble_number / n_chains) if int(ensemble_number / n_chains)>0 else decorrelate

        elif self.sampler_name == "PottsModel":
            factor_therm = 5
            factor_decorrelate = 1
            tau = L**2
            n_therm = int(factor_therm * tau)
            decorrelate = int(factor_decorrelate * L) # Note: L not L**2
            n_sweeps = int(ensemble_number / n_chains) * decorrelate
            if decorrelate == 0:
                decorrelate = 1
            if n_sweeps == 0:
                n_sweeps = decorrelate * int(ensemble_number / n_chains) if int(ensemble_number / n_chains)>0 else decorrelate
        else:
            raise ValueError(f"Unsupported sampler class: {self.sampler_name}")

        logger.info(f"--- Simulation Parameters for {self.sampler_name} ---")
        logger.info(f"Lattice size (L): {L}")
        logger.info(f"Temperature range: {T_start} to {T_end} (precision: {precision}, steps: {n_temps})")
        logger.info(f"Device: {device}")
        logger.info(f"Ensemble number per T: {ensemble_number}")
        logger.info(f"Number of chains (n_chains): {n_chains}")
        if q is not None:
            logger.info(f"Potts states (q): {q}")
        if pt_enabled:
             logger.info(f"Parallel Tempering: Enabled (interval: {pt_interval}, prob: {pt_prob})")
        logger.info(f"Calculated tau: {tau:.1f}" + (f", tau_eff: {tau_eff:.1f}" if pt_enabled else ""))
        logger.info(f"Thermalization sweeps (n_therm): {n_therm} (factor: {factor_therm})")
        logger.info(f"Decorrelation sweeps (decorrelate): {decorrelate} (factor: {factor_decorrelate})")
        logger.info(f"Production sweeps (n_sweeps): {n_sweeps}")
        logger.info("-----------------------------------------------------")

        # --- 2. Initialize Sampler ---
        common_args = {'L': L, 'T': T, 'n_chains': n_chains, 'device': torch.device(device), 'use_amp': True}
        if self.sampler_name == "XYModel":
            sampler = self.sampler_class(**common_args, pt_enabled=pt_enabled)
        elif self.sampler_name == "IsingModel":
            sampler = self.sampler_class(**common_args)
        elif self.sampler_name == "PottsModel":
            sampler = self.sampler_class(**common_args, q=q)

        # --- 3. Run Simulation ---
        logger.info("Starting Monte Carlo simulation...")
        sim_start_time = time.time()
        if self.sampler_name == "XYModel" and pt_enabled:
             samples = sampler(n_sweeps=n_sweeps, n_therm=n_therm, decorrelate=decorrelate, pt_interval=pt_interval)
        else: # Ising, Potts, or XY without PT
             samples = sampler(n_sweeps=n_sweeps, n_therm=n_therm, decorrelate=decorrelate)
        sim_end_time = time.time()
        spend_time = sim_end_time - sim_start_time
        logger.info(f"Simulation finished. Elapsed time: {spend_time:.2f} s")
        logger.info(f"Generated samples shape: {samples.shape}") # Should be (n_temps, ensemble_number, L, L)

        # --- 4. Setup Save Directories ---
        mc_data_dir = self._setup_save_subdir("mc_data")
        sampler_dir = self._setup_save_subdir(mc_data_dir / self.sampler_name)
        field_config_dir = self._setup_save_subdir(sampler_dir / "field_config")
        measurement_dir = self._setup_save_subdir(sampler_dir / "measurement")

        # --- 5. Generate Filename Base ---
        # Include key parameters for easy identification
        filename_base = f"{self.sampler_name}_L{L}_T{T_start:.3f}-{T_end:.3f}_prec{precision:.3f}_ens{ensemble_number}"
        # Replace dots with underscores for filenames if needed, though Path handles paths well.
        filename_base = filename_base.replace('.', '_')

        # --- 6. Save Field Configurations ---
        config_save_path = field_config_dir / f"{filename_base}_configs.pt"

        # Create labels tensor (Temperature, Lattice Size) for each sample configuration
        # Reshape T to match the first dimension of samples (n_temps)
        T_reshaped = T.view(-1, 1).cpu() # Move to CPU for label creation
        L_reshaped = torch.full_like(T_reshaped, float(L))

        # Concatenate along the last dimension: (n_temps, 2)
        labels = torch.cat([T_reshaped, L_reshaped], dim=1)

        logger.info(f"Saving field configurations to: {config_save_path}")
        logger.info(f"Configs tensor shape: {samples.shape}")
        logger.info(f"Labels tensor shape: {labels.shape}")
        print(f"labels shape: {labels.shape}") # Match user print
        print(f"first label pair: {labels[0]}") # Match user print

        torch.save(
            {
                'configs': samples.cpu(), # Ensure samples are on CPU for saving
                'labels': labels,
                'spend_time': spend_time,
                'params': {
                     'L': L, 'T_start': T_start, 'T_end': T_end, 'precision': precision,
                     'device': device, 'ensemble_number': ensemble_number, 'n_chains': n_chains,
                     'n_therm': n_therm, 'decorrelate': decorrelate, 'n_sweeps': n_sweeps,
                     'pt_enabled': pt_enabled, 'pt_interval': pt_interval if pt_enabled else None,
                     'pt_prob': pt_prob if pt_enabled else None, 'q': q if self.sampler_name == "PottsModel" else None
                }
            },
            config_save_path
        )
        logger.info("Field configurations saved.")

        # --- 7. Perform Measurements ---
        logger.info("Performing measurements...")
        sampler.spins = samples # Assign the generated samples back to the sampler for measurement methods

        del samples # Free memory for large tensors
        gc.collect()
        # Clear CUDA cache if using GPU
        torch.cuda.empty_cache() if torch.cuda.is_available() and device.startswith("cuda") else None

        measurements: Dict[str, Any] = {}
        temp_cpu = T.cpu().numpy() # Temperatures on CPU for plotting

        # Determine device type for AMP context
        amp_device_type = device.split(':')[0] if ':' in device else device
        if amp_device_type not in ["cuda", "mps", "cpu"]:
            amp_device_type = "cpu" # Fallback for AMP context if device is unusual

        with torch.autocast(device_type=amp_device_type):
            with torch.no_grad():
                # Common measurements
                energy = sampler.compute_average_energy()
                capacity = sampler.compute_specific_heat_capacity()
                magnetization = sampler.compute_magnetization()
                susceptibility = sampler.compute_susceptibility()

                measurements['temperature'] = temp_cpu
                measurements['energy'] = energy.cpu().numpy()
                measurements['specific_heat'] = capacity.cpu().numpy()
                measurements['magnetization'] = magnetization.cpu().numpy()
                measurements['susceptibility'] = susceptibility.cpu().numpy()

                del energy, capacity, magnetization, susceptibility # Free memory
                gc.collect()
                # Clear CUDA cache if using GPU
                torch.cuda.empty_cache() if torch.cuda.is_available() and device.startswith("cuda") else None

                # Sampler-specific measurements & plots
                plt.style.use('seaborn-v0_8-pastel')

                if self.sampler_name == "XYModel":
                    stiffness = sampler.compute_spin_stiffness()
                    vortex_density = sampler.compute_vortex_density(low_memory=True)
                    measurements['spin_stiffness'] = stiffness.cpu().numpy()
                    measurements['vortex_density'] = vortex_density.cpu().numpy()

                    del stiffness, vortex_density # Free memory
                    gc.collect()
                    # Clear CUDA cache if using GPU
                    torch.cuda.empty_cache() if torch.cuda.is_available() and device.startswith("cuda") else None

                    # Plotting
                    fig, axs = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
                    fig.suptitle(f'XY Model Measurements (L={L})', fontsize=16)
                    axs = axs.ravel() # Flatten axes array

                    tc_bkt = 0.893 # BKT transition temperature

                    axs[0].plot(temp_cpu, measurements['energy'], marker='o', linestyle='-', label='E', color='#CDB7FF')
                    axs[0].set_xlabel('Temperature (T)')
                    axs[0].set_ylabel('Energy (E)')
                    axs[0].set_title('Energy')
                    axs[0].legend()

                    axs[1].plot(temp_cpu, measurements['specific_heat'], marker='o', linestyle='-', label=r'$C_v$', color='#CDB7FF')
                    peak_cv_idx = measurements['specific_heat'].argmax()
                    axs[1].axvline(temp_cpu[peak_cv_idx], color='#FFA3A1', linestyle='--', label=f'$T_{{peak}}={temp_cpu[peak_cv_idx]:.3f}$')
                    axs[1].axvline(tc_bkt, color='#A0FFAC', linestyle='--', label=f'$T_{{BKT}}={tc_bkt:.3f}$')
                    axs[1].set_xlabel('Temperature (T)')
                    axs[1].set_ylabel('Specific Heat ($C_v$)')
                    axs[1].set_title('Specific Heat Capacity')
                    axs[1].legend()

                    axs[2].plot(temp_cpu, measurements['magnetization'], marker='o', linestyle='-', label='M', color='#CDB7FF')
                    axs[2].axvline(tc_bkt, color='#A0FFAC', linestyle='--', label=f'$T_{{BKT}}={tc_bkt:.3f}$')
                    axs[2].set_xlabel('Temperature (T)')
                    axs[2].set_ylabel('Magnetization (M)')
                    axs[2].set_title('Magnetization')
                    axs[2].legend()

                    axs[3].plot(temp_cpu, measurements['susceptibility'], marker='o', linestyle='-', label=r'$\chi$', color='#CDB7FF')
                    peak_chi_idx = measurements['susceptibility'].argmax()
                    axs[3].axvline(temp_cpu[peak_chi_idx], color='#FFA3A1', linestyle='--', label=f'$T_{{peak}}={temp_cpu[peak_chi_idx]:.3f}$')
                    axs[3].axvline(tc_bkt, color='#A0FFAC', linestyle='--', label=f'$T_{{BKT}}={tc_bkt:.3f}$')
                    axs[3].set_xlabel('Temperature (T)')
                    axs[3].set_ylabel('Susceptibility ($\\chi$)')
                    axs[3].set_title('Susceptibility')
                    axs[3].legend()

                    axs[4].plot(temp_cpu, measurements['spin_stiffness'], marker='o', linestyle='-', label=r'$\rho_s$', color='#CDB7FF')
                    axs[4].plot(temp_cpu, (2 * temp_cpu / torch.pi), linestyle='--', label='2T/π')
                    axs[4].axvline(tc_bkt, color='#A0FFAC', linestyle='--', label=f'$T_{{BKT}}={tc_bkt:.3f}$')
                    axs[4].set_xlabel('Temperature (T)')
                    axs[4].set_ylabel('Spin Stiffness ($\\rho_s$)')
                    axs[4].set_title('Spin Stiffness')
                    axs[4].legend()

                    axs[5].plot(temp_cpu, measurements['vortex_density'], marker='o', linestyle='-', label=r'$\rho_v$', color='#CDB7FF')
                    axs[5].axvline(tc_bkt, color='#A0FFAC', linestyle='--', label=f'$T_{{BKT}}={tc_bkt:.3f}$')
                    axs[5].set_xlabel('Temperature (T)')
                    axs[5].set_ylabel('Vortex Density ($\\rho_v$)')
                    axs[5].set_title('Vortex Density')
                    axs[5].legend()

                elif self.sampler_name == "IsingModel":
                    binder_cumulant = sampler.compute_binder_cumulant()
                    domain_wall_density = sampler.compute_domain_wall_density()
                    exact_magnetization = sampler.compute_exact_magnetization()
                    measurements['binder_cumulant'] = binder_cumulant.cpu().numpy()
                    measurements['domain_wall_density'] = domain_wall_density.cpu().numpy()
                    measurements['exact_magnetization'] = exact_magnetization.cpu().numpy()

                    del binder_cumulant, domain_wall_density, exact_magnetization # Free memory
                    gc.collect()
                    # Clear CUDA cache if using GPU
                    torch.cuda.empty_cache() if torch.cuda.is_available() and device.startswith("cuda") else None

                    # Plotting
                    fig, axs = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
                    fig.suptitle(f'Ising Model Measurements (L={L})', fontsize=16)
                    axs = axs.ravel()

                    tc_ising = 2 / torch.log1p(torch.sqrt(torch.tensor(2.0))) # Tc = 2 / ln(1 + sqrt(2)) ~ 2.269

                    axs[0].plot(temp_cpu, measurements['energy'], marker='o', linestyle='-', label='E', color='#CDB7FF')
                    axs[0].set_xlabel('Temperature (T)')
                    axs[0].set_ylabel('Energy (E)')
                    axs[0].set_title('Energy')
                    axs[0].legend()

                    axs[1].plot(temp_cpu, measurements['specific_heat'], marker='o', linestyle='-', label=r'$C_v$', color='#CDB7FF')
                    peak_cv_idx = measurements['specific_heat'].argmax()
                    axs[1].axvline(temp_cpu[peak_cv_idx], color='#FFA3A1', linestyle='--', label=f'$T_{{peak}}={temp_cpu[peak_cv_idx]:.3f}$')
                    axs[1].axvline(tc_ising, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_ising:.3f}$')
                    axs[1].set_xlabel('Temperature (T)')
                    axs[1].set_ylabel('Specific Heat ($C_v$)')
                    axs[1].set_title('Specific Heat Capacity')
                    axs[1].legend()

                    axs[2].plot(temp_cpu, measurements['magnetization'], marker='o', linestyle='-', label='|M|', color='#CDB7FF')
                    axs[2].plot(temp_cpu, measurements['exact_magnetization'], linestyle='--', label='$M_{exact}$')
                    axs[2].axvline(tc_ising, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_ising:.3f}$')
                    axs[2].set_xlabel('Temperature (T)')
                    axs[2].set_ylabel('Magnetization (M)')
                    axs[2].set_title('Magnetization')
                    axs[2].legend()

                    axs[3].plot(temp_cpu, measurements['susceptibility'], marker='o', linestyle='-', label=r'$\chi$', color='#CDB7FF')
                    peak_chi_idx = measurements['susceptibility'].argmax()
                    axs[3].axvline(temp_cpu[peak_chi_idx], color='#FFA3A1', linestyle='--', label=f'$T_{{peak}}={temp_cpu[peak_chi_idx]:.3f}$')
                    axs[3].axvline(tc_ising, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_ising:.3f}$')
                    axs[3].set_xlabel('Temperature (T)')
                    axs[3].set_ylabel('Susceptibility ($\\chi$)')
                    axs[3].set_title('Susceptibility')
                    axs[3].legend()

                    axs[4].plot(temp_cpu, measurements['binder_cumulant'], marker='o', linestyle='-', label='U4', color='#CDB7FF')
                    axs[4].axvline(tc_ising, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_ising:.3f}$')
                    axs[4].set_xlabel('Temperature (T)')
                    axs[4].set_ylabel('Binder Cumulant (U4)')
                    axs[4].set_title('Binder Cumulant')
                    axs[4].legend()

                    axs[5].plot(temp_cpu, measurements['domain_wall_density'], marker='o', linestyle='-', label=r'$\rho_{dw}$', color='#CDB7FF')
                    axs[5].axvline(tc_ising, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_ising:.3f}$')
                    axs[5].set_xlabel('Temperature (T)')
                    axs[5].set_ylabel('Domain Wall Density')
                    axs[5].set_title('Domain Wall Density')
                    axs[5].legend()

                elif self.sampler_name == "PottsModel":
                    binder_cumulant = sampler.compute_binder_cumulant()
                    entropy = sampler.compute_entropy()
                    measurements['binder_cumulant'] = binder_cumulant.cpu().numpy()
                    measurements['entropy'] = entropy.cpu().numpy()

                    del binder_cumulant, entropy # Free memory
                    gc.collect()
                    # Clear CUDA cache if using GPU
                    torch.cuda.empty_cache() if torch.cuda.is_available() and device.startswith("cuda") else None

                    # Plotting
                    fig, axs = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
                    fig.suptitle(f'Potts Model Measurements (L={L}, q={q})', fontsize=16)
                    axs = axs.ravel()

                    tc_potts = 1 / torch.log1p(torch.sqrt(torch.tensor(float(q)))) # Tc = 1 / ln(1 + sqrt(q))

                    axs[0].plot(temp_cpu, measurements['energy'], marker='o', linestyle='-', label='E', color='#CDB7FF')
                    axs[0].set_xlabel('Temperature (T)')
                    axs[0].set_ylabel('Energy (E)')
                    axs[0].set_title('Energy')
                    axs[0].legend()

                    axs[1].plot(temp_cpu, measurements['specific_heat'], marker='o', linestyle='-', label=r'$C_v$', color='#CDB7FF')
                    peak_cv_idx = measurements['specific_heat'].argmax()
                    axs[1].axvline(temp_cpu[peak_cv_idx], color='#FFA3A1', linestyle='--', label=f'$T_{{peak}}={temp_cpu[peak_cv_idx]:.3f}$')
                    axs[1].axvline(tc_potts, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_potts:.3f}$')
                    axs[1].set_xlabel('Temperature (T)')
                    axs[1].set_ylabel('Specific Heat ($C_v$)')
                    axs[1].set_title('Specific Heat Capacity')
                    axs[1].legend()

                    axs[2].plot(temp_cpu, measurements['magnetization'], marker='o', linestyle='-', label='M', color='#CDB7FF')
                    axs[2].axvline(tc_potts, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_potts:.3f}$')
                    axs[2].set_xlabel('Temperature (T)')
                    axs[2].set_ylabel('Magnetization (M)')
                    axs[2].set_title('Magnetization')
                    axs[2].legend()

                    axs[3].plot(temp_cpu, measurements['susceptibility'], marker='o', linestyle='-', label=r'$\chi$', color='#CDB7FF')
                    peak_chi_idx = measurements['susceptibility'].argmax()
                    axs[3].axvline(temp_cpu[peak_chi_idx], color='#FFA3A1', linestyle='--', label=f'$T_{{peak}}={temp_cpu[peak_chi_idx]:.3f}$')
                    axs[3].axvline(tc_potts, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_potts:.3f}$')
                    axs[3].set_xlabel('Temperature (T)')
                    axs[3].set_ylabel('Susceptibility ($\\chi$)')
                    axs[3].set_title('Susceptibility')
                    axs[3].legend()

                    axs[4].plot(temp_cpu, measurements['binder_cumulant'], marker='o', linestyle='-', label='U4', color='#CDB7FF')
                    axs[4].axvline(tc_potts, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_potts:.3f}$')
                    axs[4].set_xlabel('Temperature (T)')
                    axs[4].set_ylabel('Binder Cumulant (U4)')
                    axs[4].set_title('Binder Cumulant')
                    axs[4].legend()

                    axs[5].plot(temp_cpu, measurements['entropy'], marker='o', linestyle='-', label='S', color='#CDB7FF')
                    axs[5].axvline(tc_potts, color='#A0FFAC', linestyle='--', label=f'$T_c = {tc_potts:.3f}$')
                    axs[5].set_xlabel('Temperature (T)')
                    axs[5].set_ylabel('Entropy (S)')
                    axs[5].set_title('Entropy')
                    axs[5].legend()

        logger.info("Measurements complete.")

        # --- 8. Save Measurements and Plots ---
        measurement_save_path_pt = measurement_dir / f"{filename_base}_measurements.pt"
        measurement_save_path_png = measurement_dir / f"{filename_base}_plots.png"

        logger.info(f"Saving measurements dictionary to: {measurement_save_path_pt}")
        torch.save(measurements, measurement_save_path_pt)
        logger.info("Measurements dictionary saved.")

        logger.info(f"Saving plots to: {measurement_save_path_png}")
        fig.savefig(measurement_save_path_png, dpi=300)
        plt.close(fig) # Close the figure to free memory
        logger.info("Plots saved.")

        # --- 9. Cleanup ---
        del labels, measurements, sampler, fig, axs
        # Explicitly delete large tensors and objects
        if 'stiffness' in locals(): del stiffness
        if 'vortex_density' in locals(): del vortex_density
        if 'binder_cumulant' in locals(): del binder_cumulant
        if 'domain_wall_density' in locals(): del domain_wall_density
        if 'exact_magnetization' in locals(): del exact_magnetization
        if 'entropy' in locals(): del entropy
        gc.collect()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache.")

        end_total_time = time.time()
        logger.info(f"MCDataGenerator call finished. Total time: {end_total_time - start_total_time:.2f} s")


if __name__ == '__main__':

    save_directory = Path("./") # Example save directory

    # --- XY Model Example ---
    try:
        print("\n--- Running XY Model Example ---")
        xy_generator = MCDataGenerator(sampler_class=XYModel, save_dir_root=save_directory)
        xy_generator(L=8, T_start=0.1, T_end=2.0, precision=0.2, device="cuda:0" if torch.cuda.is_available() else "cpu", ensemble_number=120) # Smaller L/ensemble for quick test
        print("--- XY Model Example Finished ---")
    except Exception as e:
        logger.error(f"Error running XY Model example: {e}", exc_info=True)

    # --- Ising Model Example ---
    try:
        print("\n--- Running Ising Model Example ---")
        ising_generator = MCDataGenerator(sampler_class=IsingModel, save_dir_root=save_directory)
        ising_generator(L=8, T_start=1.0, T_end=3.5, precision=0.25, device="cuda:0" if torch.cuda.is_available() else "cpu", ensemble_number=120) # Smaller L/ensemble for quick test
        print("--- Ising Model Example Finished ---")
    except Exception as e:
         logger.error(f"Error running Ising Model example: {e}", exc_info=True)

    # --- Potts Model Example ---
    try:
        print("\n--- Running Potts Model Example ---")
        potts_generator = MCDataGenerator(sampler_class=PottsModel, save_dir_root=save_directory)
        potts_generator(L=8, T_start=0.5, T_end=1.5, precision=0.1, device="cuda:0" if torch.cuda.is_available() else "cpu", ensemble_number=120) # Smaller L/ensemble for quick test
        print("--- Potts Model Example Finished ---")
    except Exception as e:
         logger.error(f"Error running Potts Model example: {e}", exc_info=True)