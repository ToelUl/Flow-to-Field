import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal


def logit_normal_sampler(shape: tuple, sigma_logit: float = 1.0, device = 'cpu') -> torch.Tensor:
    """Generates samples from a Logit-Normal distribution centered at 0.5.

    This function uses the Logit-Normal distribution to generate samples within
    the (0, 1) interval that are concentrated around 0.5. This is achieved by:
    1. Sampling from a Normal distribution with mean 0 and standard deviation
       `sigma_logit`: `X ~ Normal(0, sigma_logit)`.
    2. Applying the sigmoid function: `Z = sigmoid(X) = 1 / (1 + exp(-X))`.

    Since sigmoid(0) = 0.5, centering the underlying Normal distribution at 0
    results in Logit-Normal samples concentrated around 0.5. The parameter
    `sigma_logit` controls the degree of concentration: smaller values lead to
    samples being more tightly clustered around 0.5.

    This approach provides a way to generate samples concentrated at 0.5 within
    (0, 1) based on the Normal/Log-Normal family of distributions.

    Args:
        shape (tuple): The desired shape of the output tensor.
        sigma_logit (float, optional): The standard deviation for the underlying
            Normal(0, sigma_logit) distribution. Must be > 0. Controls the
            concentration around 0.5 (smaller sigma -> more concentrated).
            Defaults to 1.0.
        device (str, optional): The device on which to return the sampling.

    Returns:
        torch.Tensor: A tensor of the specified shape containing samples from the
                      Logit-Normal distribution centered at 0.5.

    Raises:
        ValueError: If sigma_logit is not positive.
    """
    # --- Input Validation ---
    if not sigma_logit > 0:
        raise ValueError(f"Parameter sigma_logit must be positive, but got {sigma_logit}")

    # --- Define the base Normal distribution centered at 0 ---
    # Use 0.0 to ensure float type for the mean
    base_normal_dist = Normal(0.0, sigma_logit)

    # --- Generate samples from the Normal distribution ---
    normal_samples = base_normal_dist.sample(shape)

    # --- Apply sigmoid transformation to get Logit-Normal samples ---
    logit_normal_samples = torch.sigmoid(normal_samples)

    return logit_normal_samples.to(device)

if __name__ == "__main__":
    # --- Configuration ---
    num_samples = 1000000
    # Standard deviation for the underlying Normal(0, sigma) distribution.
    # Smaller values (e.g., 0.3) lead to higher concentration around 0.5.
    # Larger values (e.g., 1.0) lead to wider spread.
    param_sigma_logit = 1.0

    # --- Generate Samples ---
    centered_samples = logit_normal_sampler(
        (num_samples,),
        sigma_logit=param_sigma_logit
    )

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.hist(centered_samples.cpu().numpy(), bins=1000, density=True, alpha=0.75,
             color='mediumseagreen', label=f'LogitNormal(mu=0, sigma={param_sigma_logit:.2f}) Samples')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Construct the title dynamically
    title = (f'Logit-Normal Distribution Centered at 0.5\n'
             f'(Normal mean=0.0, sigma={param_sigma_logit:.2f})')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Center (0.5)') # Add vertical line at 0.5
    plt.legend() # Show legend again to include the vertical line label
    plt.tight_layout()
    plt.show()