#!/usr/bin/env python3
"""Minimal NCCL communication test between GPUs.

Usage:
    torchrun --standalone --nproc_per_node=2 scripts/test_nccl.py
"""
import os
import sys
import torch
import torch.distributed as dist
from datetime import timedelta

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        print("ERROR: Must run with torchrun (LOCAL_RANK not set)")
        sys.exit(1)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Initialized on {device}", flush=True)

    # Test 1: Simple allreduce
    print(f"[Rank {rank}] Test 1: allreduce...", flush=True)
    tensor = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert tensor.item() == expected, f"allreduce failed: got {tensor.item()}, expected {expected}"
    print(f"[Rank {rank}] Test 1 PASSED: allreduce = {tensor.item()}", flush=True)

    # Test 2: Broadcast
    print(f"[Rank {rank}] Test 2: broadcast...", flush=True)
    tensor = torch.tensor([42.0 if rank == 0 else 0.0], device=device)
    dist.broadcast(tensor, src=0)
    assert tensor.item() == 42.0, f"broadcast failed: got {tensor.item()}"
    print(f"[Rank {rank}] Test 2 PASSED: broadcast = {tensor.item()}", flush=True)

    # Test 3: Large tensor allreduce (simulate DDP param sync)
    print(f"[Rank {rank}] Test 3: large allreduce (256MB)...", flush=True)
    big_tensor = torch.randn(64_000_000, device=device)  # ~256MB in fp32
    dist.all_reduce(big_tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] Test 3 PASSED: large allreduce completed", flush=True)

    # Test 4: Simple DDP wrapping of a small model
    print(f"[Rank {rank}] Test 4: DDP wrapping...", flush=True)
    model = torch.nn.Linear(100, 10).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # Forward pass
    x = torch.randn(4, 100, device=device)
    y = ddp_model(x)
    loss = y.sum()
    loss.backward()
    print(f"[Rank {rank}] Test 4 PASSED: DDP forward+backward completed", flush=True)

    # Test 5: DDP wrapping of a larger model with find_unused_parameters
    print(f"[Rank {rank}] Test 5: DDP with find_unused_parameters...", flush=True)
    large_model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 1000),
    ).to(device)
    ddp_large = torch.nn.parallel.DistributedDataParallel(
        large_model, device_ids=[local_rank], find_unused_parameters=True
    )
    x = torch.randn(8, 1024, device=device)
    y = ddp_large(x)
    loss = y.sum()
    loss.backward()
    print(f"[Rank {rank}] Test 5 PASSED", flush=True)

    dist.barrier()
    print(f"[Rank {rank}] ALL TESTS PASSED", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
