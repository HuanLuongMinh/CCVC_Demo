import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        init_method='tcp://localhost:23456', 
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


# ---------------------------------------------------------------------------
# Launch helpers for mp.spawn on 1 or 2 GPUs (Kaggle T4x2) and on CPU (tests)
# ---------------------------------------------------------------------------
import datetime
import socket


def find_free_port():
    """Ask the OS for a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def init_ddp(rank, world_size, ports, timeout_hours=2.0):
    """Init the default process group: NCCL first, gloo if NCCL fails.

    ports: (port_nccl, port_gloo), chosen in the parent process so that every rank
    agrees on them. Returns the backend actually used.
    """
    timeout = datetime.timedelta(hours=timeout_hours)
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % ports[0],
                                rank=rank, world_size=world_size, timeout=timeout)
        return 'nccl'
    except Exception as e:  # noqa: BLE001 - any NCCL init failure falls back to gloo
        print('[ddp] rank %d: NCCL init failed (%s: %s) -> falling back to gloo' % (rank, type(e).__name__, e), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:%d' % ports[1],
                                rank=rank, world_size=world_size, timeout=timeout)
        return 'gloo'


def is_main(args):
    return args.local_rank <= 0


def barrier(args):
    if args.ddp:
        dist.barrier()


def unwrap(model):
    """Return the bare model whether or not it is wrapped by DDP."""
    return model.module if hasattr(model, 'module') else model


def all_reduce_sum(tensor, args):
    if args.ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_object(obj, args):
    if not args.ddp:
        return [obj]
    out = [None] * args.world_size
    dist.all_gather_object(out, obj)
    return out


def broadcast_object(obj, args):
    """Broadcast a picklable object from rank 0 to every rank."""
    if not args.ddp:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]
