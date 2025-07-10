import torch
import matplotlib.pyplot as plt


def timestep_scheduler(t: torch.Tensor, mu: float = 0.0, sigma: float = 1.0, upper_bound: float = 1.0) -> torch.Tensor:
    """
    Remaps a 1D time tensor 't' using the formula: t' = upper_bound * exp(mu) / (exp(mu) + (upper_bound/t - 1)**sigma).

    This function applies a non-linear transformation to the input tensor,
    often used for creating non-linear schedules (e.g., in diffusion models).
    The transformation is controlled by the parameters 'mu' and 'sigma', which
    adjust the curvature of the mapping. The output tensor 't_prime' will
    typically be in the range [0, 1], depending on the input tensor 't'.

    Args:
        t (torch.Tensor): A 1-dimensional input tensor, typically containing
                          values within the range [0, 1].
        mu (float, optional): A parameter that controls the mean of the
                            transformation. Defaults to 0.0.
        sigma (float, optional): A parameter that controls the standard deviation
                            of the transformation. Defaults to 1.0.
        upper_bound (float, optional): The upper bound for the transformation,
                            typically set to 1.0. This parameter scales the
                            output tensor to ensure it remains within a desired range.

    Returns:
        torch.Tensor: The transformed 1-dimensional tensor t'.

    Raises:
        ValueError: If the input tensor 't' is not 1-dimensional.
        ValueError: If the denominator in the transformation becomes zero,
                    which would lead to division by zero.
    """
    # Input validation: Ensure 't' is a 1D tensor.
    if t.dim() != 1:
        raise ValueError("Input tensor 't' must be 1-dimensional.")

    # Calculate the denominator
    denominator = (torch.exp(mu * torch.ones_like(t)) + (upper_bound/t - 1)**sigma)

    if torch.any(denominator == 0):
        raise ValueError("Division by zero encountered in time_scheduler.")

    # Apply the transformation formula
    t_prime = upper_bound * torch.exp(mu * torch.ones_like(t)) / denominator

    return t_prime

# --- Example Usage and Visualization ---

if __name__ == "__main__":
    # 1. Define parameters for the example
    start_val: float = 0.0       # Start value for the input tensor
    end_val: float = 1.0         # End value for the input tensor
    num_steps: int = 5           # Number of points in the tensor
    mu: float = -0.3             # Mean parameter for the transformation
    sigma: float = 1.0           # Standard deviation parameter for the transformation

    # 2. Generate the original input tensor (linearly spaced)
    t_input = torch.linspace(start_val, end_val, num_steps)

    # 3. Apply the time scheduler function
    t_output = timestep_scheduler(t_input, mu, sigma)

    # 4. Visualize the input and output tensors using vertical lines
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(f"Time Scheduler Transformation (mu={mu}, sigma={sigma})", fontsize=14)

    # Plot the original tensor 't'
    axes[0].vlines(t_input.numpy(), ymin=0, ymax=1, color='blue', linestyle='-', lw=2,)
    axes[0].set_title(f'Original Linear Tensor (t) - {num_steps} steps')
    axes[0].set_yticks([])  # Hide y-axis ticks as they are not meaningful here
    axes[0].set_ylim(-0.1, 1.1) # Add padding to y-axis
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)

    # Plot the transformed tensor 't_prime'
    axes[1].vlines(t_output.numpy(), ymin=0, ymax=1, color='red', linestyle='-', lw=2,)
    axes[1].set_title(f'Transformed Tensor (t\') using time_scheduler')
    axes[1].set_yticks([]) # Hide y-axis ticks
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    # Set common x-axis label
    plt.xlabel("Value")

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the plot
    plt.show()

    # 5. Print the numerical values (optional)
    print("-" * 30)
    print("Original Input Tensor (t):")
    print(t_input)
    print("-" * 30)
    print(f"Transformed Tensor (t') with mu={mu}, sigma={sigma}:")
    print(t_output)
    print("-" * 30)