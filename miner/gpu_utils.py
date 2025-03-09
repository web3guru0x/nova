# gpu_utils.py
import os
import torch
import gc
import time
import contextlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gpu_utils')

def setup_gpu_for_h200():
    """
    Configure the environment for optimal H200 SXM GPU performance.
    """
    # Ensure we're using CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available! GPU optimizations will not work.")
        return False

    # Print GPU information
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} GPU devices")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_cap = torch.cuda.get_device_capability(i)
        logger.info(f"Device {i}: {device_name}, Capability: {device_cap}")
    
    # Enable TF32 precision (faster and almost as accurate as FP32 on Ampere+ GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
    
    # Disable cuDNN determinism for better performance
    torch.backends.cudnn.deterministic = False
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    logger.info("GPU configured for optimal H200 SXM performance")
    return True

def empty_gpu_cache(wait=True):
    """
    Empty GPU cache and run garbage collection.
    
    Args:
        wait (bool): If True, wait for all CUDA operations to finish first
    """
    if wait:
        torch.cuda.synchronize()
    
    # Run Python garbage collection first
    gc.collect()
    
    # Then empty CUDA cache
    torch.cuda.empty_cache()
    
    # For newer PyTorch versions that support it
    if hasattr(torch.cuda.memory, 'empty_cache'):
        torch.cuda.memory.empty_cache()

@contextlib.contextmanager
def gpu_timer(name="Operation"):
    """
    Context manager for timing GPU operations.
    
    Args:
        name (str): Name of the operation for logging
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000  # Convert to seconds
        logger.info(f"{name} completed in {elapsed_time:.4f} seconds")

def get_gpu_memory_usage():
    """
    Get current GPU memory usage information.
    
    Returns:
        dict: Dictionary with memory usage statistics
    """
    memory_stats = {}
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # Convert to GB
        
        memory_stats[f"cuda:{i}"] = {
            "allocated_GB": allocated,
            "reserved_GB": reserved,
            "utilization_percent": torch.cuda.utilization(i),
        }
    
    return memory_stats

def optimize_memory_for_inference(max_batch_size=None):
    """
    Optimize GPU memory configuration for inference workloads.
    
    Args:
        max_batch_size (int, optional): Maximum expected batch size
    """
    # Free all unused memory
    empty_gpu_cache()
    
    # For large memory GPUs like H200, reserve some memory to prevent fragmentation
    if max_batch_size is not None:
        dummy_tensors = []
        try:
            # Reserve memory for the largest expected batch size
            dummy_tensors.append(torch.zeros(max_batch_size, 1024, 1024, device='cuda'))
            logger.info(f"Reserved memory for batch size {max_batch_size}")
            # Clear immediately to make the reserved memory available
            dummy_tensors.clear()
        except Exception as e:
            logger.warning(f"Failed to reserve memory: {e}")
    
    # Set PyTorch to release memory more aggressively
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'
    
    logger.info("GPU memory optimized for inference")