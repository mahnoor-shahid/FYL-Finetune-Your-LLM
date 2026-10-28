import torch


def get_device(verbose=True):
    """
    Return the best available device: CUDA GPU, Apple MPS, or CPU.

    The Trainer picks the device on its own; use this for anything outside it
    (e.g. model.to(device) and moving inputs for your own inference code).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print_device_info(device)
    return device


def print_device_info(device):
    """Print which device is used, and GPU details when on CUDA."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"✅ Using GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB, "
              f"CUDA {torch.version.cuda}, PyTorch {torch.__version__})")
    elif device.type == "mps":
        print("✅ Using Apple GPU (MPS)")
    else:
        hint = ("PyTorch is a CPU-only build" if torch.version.cuda is None
                else f"PyTorch has CUDA {torch.version.cuda} but no usable GPU/driver was found")
        print(f"⚠️ Using CPU ({hint}); training will be slow")


def get_precision_flags():
    """
    Mixed-precision settings for TrainingArguments: bf16 on GPUs that support it
    (RTX 30-series and newer), fp16 on older GPUs, full precision on CPU/MPS.

    Usage: TrainingArguments(..., **get_precision_flags())
    """
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}
