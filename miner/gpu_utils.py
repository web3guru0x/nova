# gpu_utils.py
import os
import torch
import gc
import time
import contextlib
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gpu_utils')

def setup_gpu_for_h200():
    """
    Configure the environment for optimal H200 SXM GPU performance with 141GB VRAM.
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
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
        logger.info(f"Device {i}: {device_name}, Capability: {device_cap}, Memory: {total_mem:.2f} GB")
    
    # Enable TF32 precision (faster and almost as accurate as FP32 on Ampere+ GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
    
    # Disable cuDNN determinism for better performance
    torch.backends.cudnn.deterministic = False
    
    # Set memory allocation strategy - increased chunk size for better performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,garbage_collection_threshold:0.9'
    
    # Preallocate memory to reduce fragmentation - larger for H200's 141GB VRAM
    try:
        dummy = torch.zeros(4096, 4096, device="cuda")
        del dummy
        torch.cuda.empty_cache()
        logger.info("Preallocated GPU memory to reduce fragmentation")
    except Exception as e:
        logger.warning(f"Failed to preallocate memory: {e}")
    
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

def create_parallel_streams(num_streams=4):
    """
    Create multiple CUDA streams for parallel execution.
    
    Args:
        num_streams (int): Number of CUDA streams to create
        
    Returns:
        list: List of CUDA stream objects
    """
    return [torch.cuda.Stream() for _ in range(num_streams)]

@contextlib.contextmanager
def gpu_timer(name="Operation", threshold_ms=None):
    """
    Context manager for timing GPU operations.
    
    Args:
        name (str): Name of the operation for logging
        threshold_ms (float, optional): Log as warning if operation exceeds this time in ms
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)
        elapsed_time_s = elapsed_time_ms / 1000  # Convert to seconds
        
        if threshold_ms and elapsed_time_ms > threshold_ms:
            logger.warning(f"{name} took {elapsed_time_s:.4f} seconds (exceeds threshold of {threshold_ms/1000:.4f}s)")
        else:
            logger.info(f"{name} completed in {elapsed_time_s:.4f} seconds")
        
        return elapsed_time_s

def get_gpu_memory_usage(detailed=False):
    """
    Get current GPU memory usage information.
    
    Args:
        detailed (bool): If True, return detailed memory stats
        
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
        
        if detailed and hasattr(torch.cuda, 'memory_stats'):
            memory_stats[f"cuda:{i}"]["detailed"] = torch.cuda.memory_stats(i)
    
    return memory_stats

def optimize_memory_for_inference(max_batch_size=None, num_warmup_iterations=2):
    """
    Optimize GPU memory configuration for inference workloads.
    
    Args:
        max_batch_size (int, optional): Maximum expected batch size
        num_warmup_iterations (int): Number of warmup iterations to perform
    """
    # Free all unused memory
    empty_gpu_cache()
    
    # For large memory GPUs like H200, reserve some memory to prevent fragmentation
    if max_batch_size is not None:
        try:
            # Reserve memory for the largest expected batch size
            dummy_tensor_size = min(max_batch_size, 4096)  # Cap at 4096 to avoid OOM
            dummy_tensors = []
            
            # Create a mix of tensor sizes to better simulate real workloads
            dummy_tensors.append(torch.zeros(dummy_tensor_size, 1024, device='cuda'))
            dummy_tensors.append(torch.zeros(dummy_tensor_size//2, 2048, device='cuda'))
            dummy_tensors.append(torch.zeros(dummy_tensor_size//4, 4096, device='cuda'))
            
            logger.info(f"Reserved memory for batch size {max_batch_size}")
            
            # Perform warmup iterations to stabilize GPU memory allocation
            for i in range(num_warmup_iterations):
                # Simulate computation
                result = torch.matmul(dummy_tensors[0], dummy_tensors[0].transpose(0, 1))
                result = torch.nn.functional.relu(result)
                torch.cuda.synchronize()
            
            # Clear immediately to make the reserved memory available
            for tensor in dummy_tensors:
                del tensor
            dummy_tensors = []
            
        except Exception as e:
            logger.warning(f"Failed to reserve memory: {e}")
    
    # Set PyTorch to release memory more aggressively
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,garbage_collection_threshold:0.9'
    
    # Run garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info("GPU memory optimized for inference")

def track_gpu_memory_usage(label="Current"):
    """Track and return detailed GPU memory usage with a label"""
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    # Get detailed stats if available
    detailed_stats = {}
    if hasattr(torch.cuda, 'memory_stats'):
        stats = torch.cuda.memory_stats()
        detailed_stats = {
            'active_bytes': stats.get('active_bytes.all.current', 0) / (1024 ** 3),
            'inactive_split_bytes': stats.get('inactive_split_bytes.all.current', 0) / (1024 ** 3),
            'reserved_bytes': stats.get('reserved_bytes.all.current', 0) / (1024 ** 3),
            'active_count': stats.get('active.all.current', 0),
            'alloc_count': stats.get('allocation.all.current', 0)
        }
    
    # Get utilization if available
    utilization = torch.cuda.utilization()
    
    # Log the statistics
    logger.info(f"GPU MEMORY [{label}]: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max, {utilization}% utilization")
    
    if detailed_stats:
        logger.info(f"GPU MEMORY DETAILS [{label}]: " + 
                   f"Active: {detailed_stats['active_bytes']:.2f}GB ({detailed_stats['active_count']} blocks), " +
                   f"Inactive: {detailed_stats['inactive_split_bytes']:.2f}GB, " +
                   f"Allocations: {detailed_stats['alloc_count']}")
    
    # Return statistics dictionary
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
        'utilization': utilization,
        'detailed': detailed_stats if detailed_stats else None
    }

class CudaStreamPool:
    """Pool of CUDA streams for parallel execution"""
    
    def __init__(self, num_streams=4):
        """
        Initialize a pool of CUDA streams for parallel processing
        
        Args:
            num_streams (int): Number of CUDA streams to create
        """
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_idx = 0
        self.num_streams = num_streams
        
    def get_next_stream(self):
        """Get the next available stream in round-robin fashion"""
        stream = self.streams[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.num_streams
        return stream
    
    def synchronize_all(self):
        """Wait for all streams to complete"""
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

class BatchProcessor:
    """Manages parallel batch processing using CUDA streams"""
    
    def __init__(self, batch_size=4096, num_streams=4):
        """
        Initialize a batch processor for parallel GPU processing
        
        Args:
            batch_size (int): Size of batches to process
            num_streams (int): Number of CUDA streams to use
        """
        self.batch_size = batch_size
        self.stream_pool = CudaStreamPool(num_streams)
        self.pending_batches = []
        self.results = []
        
    def add_batch(self, batch_data, process_fn):
        """
        Add a batch for processing with the given function
        
        Args:
            batch_data: The batch data to process
            process_fn: Function to process the batch data
        """
        stream = self.stream_pool.get_next_stream()
        with torch.cuda.stream(stream):
            result = process_fn(batch_data)
            self.pending_batches.append((stream, result))
        
    def collect_ready_results(self, wait_all=False):
        """
        Collect results from completed batches
        
        Args:
            wait_all (bool): If True, wait for all pending batches to complete
        
        Returns:
            list: List of completed batch results
        """
        if wait_all:
            self.stream_pool.synchronize_all()
            
        # Move completed batches to results
        completed = []
        remaining = []
        
        for stream, result in self.pending_batches:
            if stream.query() or wait_all:
                completed.append(result)
            else:
                remaining.append((stream, result))
                
        self.pending_batches = remaining
        self.results.extend(completed)
        
        return completed
    
    def get_all_results(self):
        """Get all completed results and wait for any pending batches"""
        self.collect_ready_results(wait_all=True)
        return self.results