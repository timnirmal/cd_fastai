import torch

# check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Availability: {cuda_available}")

# if CUDA is available, get the device count and the current device
if cuda_available:
    cuda_device_count = torch.cuda.device_count()
    current_cuda_device = torch.cuda.current_device()
    cuda_device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device Count: {cuda_device_count}")
    print(f"Current CUDA Device: {current_cuda_device} ({cuda_device_name})")

    # memory information
    print("\nMemory Usage:")
    print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
    print(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

# check for cuDNN availability and version
cudnn_available = torch.backends.cudnn.is_available()
cudnn_version = torch.backends.cudnn.version() if cudnn_available else "N/A"
print(f"\ncuDNN Availability: {cudnn_available}")
print(f"cuDNN Version: {cudnn_version}")

