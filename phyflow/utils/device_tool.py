import torch
import platform
import subprocess


def get_complete_device_info():
    """
    Prints complete device and firmware information, including OS, CPU, GPU,
    PyTorch version, CUDA version, and cuDNN version.
    """
    print("=" * 40)
    print("      Complete Device and Firmware Information")
    print("=" * 40)

    # 1. Operating System Information
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")

    # 2. CPU Information
    # Use platform to get basic info
    print(f"CPU Model: {platform.processor()}")
    # On Linux, lscpu provides more detailed information
    if platform.system() == "Linux":
        try:
            lscpu_output = subprocess.check_output("lscpu", shell=True).strip().decode()
            for line in lscpu_output.split("\n"):
                if "Model name" in line:
                    cpu_model_name = line.split(":")[1].strip()
                    print(f"CPU Full Model: {cpu_model_name}")
                    break
        except Exception as e:
            print(f"Could not get detailed CPU model from lscpu: {e}")
    # On Windows, use wmic
    elif platform.system() == "Windows":
        try:
            cpu_model_name = subprocess.check_output("wmic cpu get name", shell=True).strip().decode().split('\n')[1]
            print(f"CPU Full Model: {cpu_model_name}")
        except Exception as e:
            print(f"Could not get detailed CPU model from wmic: {e}")

    # 3. PyTorch Version
    print(f"PyTorch Version: {torch.__version__}")

    # 4. CUDA & GPU Information
    if torch.cuda.is_available():
        print("CUDA Status: \033[92mAvailable\033[0m")  # Green text for available

        # 4.1 CUDA version used by PyTorch
        print(f"CUDA Version (used by PyTorch): {torch.version.cuda}")

        # 4.2 Number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs Available: {gpu_count}")

        # 4.3 Print details for each GPU
        for i in range(gpu_count):
            print(f"--- GPU {i} ---")
            print(f"  GPU Model: {torch.cuda.get_device_name(i)}")

            # Display memory info
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated_mem = torch.cuda.memory_allocated(i) / 1e9
            cached_mem = torch.cuda.memory_reserved(i) / 1e9
            print(f"  Total Memory: {total_mem:.2f} GB")
            print(f"  Allocated Memory: {allocated_mem:.2f} GB")
            print(f"  Cached Memory (Reserved): {cached_mem:.2f} GB")

    else:
        print("CUDA Status: \033[91mNot Available\033[0m")  # Red text for not available

    # 5. cuDNN Version
    if torch.backends.cudnn.is_available():
        # torch.backends.cudnn.version() returns an integer, needs manual formatting
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN Version: {cudnn_version}")
    else:
        print("cuDNN Status: Not Available")

    print("=" * 40)


if __name__ == '__main__':
    get_complete_device_info()