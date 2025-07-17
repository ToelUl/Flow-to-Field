import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal


def logit_normal_sampler(
        shape: tuple,
        mu_logit: float = 0.0,
        sigma_logit: float = 1.0,
        device: str = 'cpu') -> torch.Tensor:
    """Generates samples from a Logit-Normal distribution.

    This function produces samples in the (0, 1) interval by transforming
    samples from a Normal distribution. The process is as follows:
    1. Sample from a Normal distribution: `X ~ Normal(mu_logit, sigma_logit)`.
    2. Apply the standard sigmoid function: `Z = sigmoid(X) = 1 / (1 + exp(-X))`.

    The `mu_logit` parameter controls the mean of the underlying Normal
    distribution, which determines the location of the mode (peak) of the
    Logit-Normal samples. The center of the distribution is `sigmoid(mu_logit)`.
    For example, a `mu_logit` of 0 centers the distribution around 0.5.

    The `sigma_logit` parameter controls the standard deviation of the Normal
    distribution, which dictates how concentrated the samples are around the
    center. Smaller values result in a tighter clustering.

    Args:
        shape (tuple): The desired shape of the output tensor.
        mu_logit (float, optional): The mean of the underlying Normal
            distribution. This controls the center of the Logit-Normal
            distribution. Defaults to 0.0.
        sigma_logit (float, optional): The standard deviation for the underlying
            Normal distribution. Must be > 0. Controls the concentration of
            samples. Defaults to 1.0.
        device (str, optional): The device on which to create the tensor.
            Defaults to 'cpu'.

    Returns:
        torch.Tensor: A tensor of the specified shape containing samples from the
                      Logit-Normal distribution.

    Raises:
        ValueError: If sigma_logit is not positive.
    """
    # --- Input Validation ---
    if not sigma_logit > 0:
        raise ValueError(f"Parameter sigma_logit must be positive, but got {sigma_logit}")

    # --- Define the base Normal distribution ---
    base_normal_dist = Normal(mu_logit, sigma_logit)

    # --- Generate samples from the Normal distribution ---
    normal_samples = base_normal_dist.sample(shape)

    # --- Apply sigmoid transformation to get Logit-Normal samples ---
    logit_normal_samples = torch.sigmoid(normal_samples)

    return logit_normal_samples.to(device)

if __name__ == "__main__":
    # --- Configuration ---
    num_samples = 1000000
    # The mean of the underlying Normal distribution.
    # This determines the center of the Logit-Normal samples.
    # e.g., mu_logit=0.0 -> center=0.5
    # e.g., mu_logit=2.0 -> center=sigmoid(2.0) approx 0.88
    param_mu_logit = -1.1
    # The standard deviation of the underlying Normal distribution.
    # Smaller values lead to higher concentration around the center.
    param_sigma_logit = 1.0

    # --- Calculate Distribution Center ---
    center_value = torch.sigmoid(torch.tensor(param_mu_logit)).item()

    # --- Generate Samples ---
    samples = logit_normal_sampler(
        (num_samples,),
        mu_logit=param_mu_logit,
        sigma_logit=param_sigma_logit
    )

    # --- Visualization ---
    plt.figure(figsize=(12, 7))
    plt.hist(samples.cpu().numpy(), bins=1000, density=True, alpha=0.75,
             color='mediumseagreen')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Construct the title dynamically
    title = (f'Logit-Normal Distribution\n'
             f'Center = sigmoid({param_mu_logit:.1f}) ≈ {center_value:.2f}, '
             f'Underlying Normal(mean={param_mu_logit:.1f}, sigma={param_sigma_logit:.1f})')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a vertical line at the calculated center
    plt.axvline(center_value, color='red', linestyle='--', linewidth=1.5,
                label=f'Center ≈ {center_value:.2f}')
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(0, 1) # Samples are within (0, 1)
    plt.tight_layout()
    plt.show()