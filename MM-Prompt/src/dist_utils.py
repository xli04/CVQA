# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Distributed training utilities for multi-GPU/multi-node training.
Provides communication primitives and helper functions for synchronized 
distributed deep learning training.
"""

import functools
import logging
import numpy as np
import pickle
import torch
import torch.distributed as dist

import torch

_LOCAL_PROCESS_GROUP = None
"""
Process group containing only processes on the same machine as the current process.
Set during initialization by 'launch()' in engine/launch.py.
"""


def get_world_size() -> int:
    """
    Get total number of distributed processes across all nodes.
    
    Returns:
        Number of processes in the distributed training (1 if not distributed)
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get global rank of current process among all distributed processes.
    
    Returns:
        Integer rank of current process (0 if not distributed)
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Get rank of current process within the local machine.
    
    Returns:
        Local rank of process within current machine (0 if not distributed)
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Get number of processes on current machine.
    
    Returns:
        Number of processes on current machine (1 if not distributed)
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).
    
    Returns:
        True if current process is the main process
    """
    return get_rank() == 0


def synchronize():
    """
    Synchronize all distributed processes (barrier).
    
    Creates a synchronization point where all processes must reach
    before continuing execution. No-op if not in distributed mode.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Get a process group based on gloo backend for all ranks.
    
    Returns:
        Process group for collective operations
        
    Note:
        Result is cached for efficiency.
        Gloo backend is used for CPU tensors when NCCL is the primary backend.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    """
    Serialize arbitrary data to a Torch tensor for distributed communication.
    
    Args:
        data: Any picklable Python object
        group: Process group for determining the correct device
        
    Returns:
        tensor: Serialized data as a ByteTensor on appropriate device
    """
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Pad tensor to the size of the largest tensor across all processes.
    
    Args:
        tensor: Input tensor to be padded
        group: Process group for the collective operation
        
    Returns:
        tuple: (list of sizes of tensor in each rank, padded tensor)
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # Pad tensor for uniform shape across processes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Gather arbitrary data from all processes in a distributed manner.
    
    Args:
        data: Any picklable Python object
        group: Process group for the collective operation (defaults to global group)
        
    Returns:
        list: List of data from all ranks in the order of their ranks
        
    Note:
        Efficiently handles non-tensor data using pickle serialization
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # Receive tensors from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Gather arbitrary data from all processes to a specific destination rank.
    
    Args:
        data: Any picklable Python object
        dst: Destination rank where data will be gathered
        group: Process group for the collective operation (defaults to global group)
        
    Returns:
        list: On destination rank, a list of data from all ranks.
              On other ranks, an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # Receiving process gets data from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Generate a random seed that is the same across all distributed processes.
    
    Returns:
        int: A random number that is identical across all workers
        
    Note:
        All workers must call this function, otherwise it will deadlock.
        Useful for ensuring reproducibility in distributed training.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary values from all processes to the main process.
    
    Args:
        input_dict: Dictionary with CUDA tensor values to be reduced
        average: If True, average the values; otherwise, sum them
        
    Returns:
        Dictionary with the same keys but reduced values
        
    Note:
        Results only valid on the main process (rank 0).
        All input tensors must be on the same device.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # Sort the keys to ensure that all processes reduce in the same order
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # Average only on the main process
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
