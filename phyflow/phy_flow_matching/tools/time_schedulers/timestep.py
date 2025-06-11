import torch
import matplotlib.pyplot as plt

# --- Function Definition ---

def timestep_scheduler(t: torch.Tensor, a: float = 0.3) -> torch.Tensor:
    """
    Remaps a 1D time tensor 't' using the formula: t' = a*t / (1 + (a - 1)*t).

    This function applies a non-linear transformation to the input tensor,
    often used for creating non-linear schedules (e.g., in diffusion models).
    The parameter 'a' controls the curvature of the mapping.

    Args:
        t (torch.Tensor): A 1-dimensional input tensor, typically containing
                          values within the range [0, 1].
        a (float): The control parameter for the mapping curvature.
                   - a > 1: Compresses points near the start (t=0) and expands
                            points near the end (t=1).
                   - a = 1: Results in a linear mapping (t' = t).
                   - 0 < a < 1: Expands points near the start and compresses
                                points near the end.
                   - a <= 0: Can lead to division by zero or non-monotonic
                             behavior; generally not recommended for standard
                             scheduling tasks.
                    Default is 0.3.

    Returns:
        torch.Tensor: The transformed 1-dimensional tensor t'.

    Raises:
        ValueError: If the input tensor 't' is not 1-dimensional.
        RuntimeWarning: May be implicitly raised by PyTorch or explicitly printed
                        if the denominator `1 + (a - 1) * t` becomes zero during
                        computation, leading to `inf` or `nan` in the output.
    """
    # Input validation: Ensure 't' is a 1D tensor.
    if t.dim() != 1:
        raise ValueError("Input tensor 't' must be 1-dimensional.")

    # Calculate the denominator
    denominator = 1.0 + (a - 1.0) * t

    if torch.any(denominator == 0):
        raise ValueError("Division by zero encountered in time_scheduler.")

    # Apply the transformation formula
    t_prime = (a * t) / denominator

    return t_prime

# --- Example Usage and Visualization ---

if __name__ == "__main__":
    # 1. Define parameters for the example
    start_val: float = 0.0       # Start value for the input tensor
    end_val: float = 1.0         # End value for the input tensor
    num_steps: int = 15          # Number of points in the tensor
    a_parameter: float = 0.3     # Control parameter 'a' (try values like 0.2, 1.0, 5.0)

    # 2. Generate the original input tensor (linearly spaced)
    t_input = torch.linspace(start_val, end_val, num_steps)

    # 3. Apply the time scheduler function
    t_output = timestep_scheduler(t_input, a_parameter)

    # 4. Visualize the input and output tensors using vertical lines
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(f"Time Scheduler Transformation (a={a_parameter})", fontsize=14)

    # Plot the original tensor 't'
    axes[0].vlines(t_input.numpy(), ymin=0, ymax=1, color='blue', linestyle='-', lw=2, label='Original t')
    axes[0].set_title(f'Original Linear Tensor (t) - {num_steps} steps')
    axes[0].set_yticks([])  # Hide y-axis ticks as they are not meaningful here
    axes[0].set_ylim(-0.1, 1.1) # Add padding to y-axis
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper left')

    # Plot the transformed tensor 't_prime'
    axes[1].vlines(t_output.numpy(), ymin=0, ymax=1, color='red', linestyle='-', lw=2, label=f'Transformed t\' (a={a_parameter})')
    axes[1].set_title(f'Transformed Tensor (t\') using time_scheduler')
    axes[1].set_yticks([]) # Hide y-axis ticks
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper left')

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
    print(f"Transformed Tensor (t') with a={a_parameter}:")
    print(t_output)
    print("-" * 30)