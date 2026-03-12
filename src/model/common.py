# src/model/common.py

import torch.distributed as dist

def init_parallel(tp_size, backend="nccl"):
    dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % tp_size != 0:
        raise ValueError(f"World size ({world_size}) must be divisible by TP size ({tp_size}).")

    dp_size = world_size // tp_size

    # ----- TP group -----
    tp_group = None
    for i in range(dp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)

        if rank in ranks:
            tp_group = group

    # ----- DP group -----
    dp_group = None
    for i in range(tp_size):
        ranks = list(range(i, world_size, tp_size))
        group = dist.new_group(ranks)

        if rank in ranks:
            dp_group = group

    return tp_group, dp_group