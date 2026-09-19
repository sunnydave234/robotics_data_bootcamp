#!/usr/bin/env python3
"""
ray_train_stub.py -- Week 3 Day 3 Mini Project (reference implementation)
==========================================================================
A CPU-runnable Ray Train (V2) loop over lerobot/aloha_mobile_cabinet's
tabular columns: trivial nn.Linear(14, 14) mapping observation.state ->
action. The model is deliberately uninteresting. What this file exists to
prove -- each piece first shown alone on a toy in Labs 0-3 -- is the four
things Ray Train owns and how to use each one correctly:

  1. DDP wiring          prepare_model()          -- wraps ONLY at world_size > 1
  2. dataset sharding    get_dataset_shard()      -- streaming_split(equal=True)
  3. checkpoint persist  report(checkpoint=...)   -- to {storage_path}/{name}
  4. restart             get_checkpoint()         -- repopulated by the SAME
                                                     (storage_path, name) pair

Resume is tested, not assumed: --kill-at-epoch E makes rank 0 os._exit(1)
right after epoch E's checkpoint is reported. Rerun the identical command
and the loop should print "RESUMED at epoch E+1".

STATUS: CONFIRMED against a real Ray install (Ray >= 2.47, V2 default) via
Labs 0-3 -- 21-row sharding split (equal=True, drops the odd row out),
Lab 2's real batch shapes, and Lab 3's real save/crash/resume cycle. One
fix folded back in from that real run: a plain function passed as
`collate_fn` prints `RayDeprecationWarning: ... deprecated in Ray 2.47`;
the fix below uses a `NumpyBatchCollateFn` subclass instead, confirmed
against Ray's own `iter_torch_batches` API doc.

Sources for every non-obvious line (all verified against Ray 2.56-era docs):
  - checkpoint guide   docs.ray.io/en/latest/train/user-guides/checkpoints.html
  - fault tolerance    docs.ray.io/en/latest/train/user-guides/fault-tolerance.html
  - prepare_model V2   docs.ray.io/en/latest/_modules/ray/train/v2/torch/train_loop_utils.html
  - DataConfig source  docs.ray.io/en/latest/_modules/ray/train/_internal/data_config.html
  - collate_fn classes docs.ray.io/en/latest/data/api/doc/ray.data.DataIterator.iter_torch_batches.html
  - Anyscale pi0.5 reference loop (Feb 10 2026)
      anyscale.com/blog/vision-language-action-pipelines-vla-robotics-ray-anyscale
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import ray
import ray.train
import ray.train.torch
from ray.data.collate_fn import NumpyBatchCollateFn

from tensor_utils import to_2d_float_array

DEFAULT_ROOT = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
STATE_DIM = 14
ACTION_DIM = 14
CKPT_FILE = "state.pt"


# ---------------------------------------------------------------------------
# Batch conversion. A callable CLASS, not a bare function -- Ray needs the
# class identity (via isinstance) to know this collate_fn expects the numpy
# batch format, not pyarrow or pandas. Passing a plain function here is
# deprecated as of Ray 2.47 (confirmed by the RayDeprecationWarning it
# actually prints). Passing this collate_fn at all switches Ray back to
# batch_format="numpy" internally, which is exactly where Day 1's
# object-dtype gotcha can reappear -- hence to_2d_float_array below, not a
# bare torch.as_tensor(batch[...]).
# ---------------------------------------------------------------------------
class StateActionCollateFn(NumpyBatchCollateFn):
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "state": torch.as_tensor(to_2d_float_array(batch, "observation.state", STATE_DIM)),
            "action": torch.as_tensor(to_2d_float_array(batch, "action", ACTION_DIM)),
        }


# ---------------------------------------------------------------------------
# The Debugging Challenge fix. prepare_model() wraps in DDP only when
# world_size > 1 (V2 source: `if parallel_strategy and world_size > 1`).
# So `model.module` exists at 2 workers and does not at 1, and a plain
# `model.state_dict()` at 2 workers carries "module."-prefixed keys that a
# fresh, unwrapped model refuses to load on resume. Unwrap conditionally.
# ---------------------------------------------------------------------------
def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def train_loop_per_worker(config: dict) -> None:
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()

    # ---- 1. build model, then restore state BEFORE wrapping (docs' order) ----
    model = nn.Linear(STATE_DIM, ACTION_DIM)
    start_epoch, opt_state = 0, None
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            state = torch.load(os.path.join(d, CKPT_FILE), map_location="cpu")
            model.load_state_dict(state["model"])
            opt_state = state["opt"]
            start_epoch = state["epoch"] + 1
        print(f"[rank {rank}] RESUMED at epoch {start_epoch}")

    model = ray.train.torch.prepare_model(model)
    # Optimizer AFTER prepare_model: on GPU, .to(device) creates new parameter
    # tensors and an optimizer built earlier would hold the old ones.
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    # ---- 2. this worker's shard: streaming_split(world, equal=True) ----
    shard = ray.train.get_dataset_shard("train")
    collate_fn = StateActionCollateFn()  # instance, not the class -- Ray checks isinstance()

    for epoch in range(start_epoch, config["epochs"]):
        t0 = time.time()
        loss_sum, n_batches, n_rows = 0.0, 0, 0
        first_loss = None
        for batch in shard.iter_torch_batches(
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            local_shuffle_buffer_size=config["shuffle_buffer"] or None,
        ):
            pred = model(batch["state"])
            loss = nn.functional.mse_loss(pred, batch["action"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            v = loss.item()
            first_loss = v if first_loss is None else first_loss
            loss_sum += v
            n_batches += 1
            n_rows += batch["state"].shape[0]

        metrics = {
            "epoch": epoch,
            "rank": rank,
            "rows_seen": n_rows,
            "first_loss": first_loss,
            "mean_loss": loss_sum / max(n_batches, 1),
            "epoch_s": round(time.time() - t0, 2),
        }
        # Only rank 0's metrics come back in result.metrics -- print every rank's.
        print(f"[rank {rank}/{world}] {metrics}")

        # ---- 3. checkpoint: EVERY rank calls report() (it's a barrier),
        #         only rank 0 attaches the payload (DDP: identical weights) ----
        with tempfile.TemporaryDirectory() as d:
            checkpoint = None
            if rank == 0:
                torch.save(
                    {"model": unwrap(model).state_dict(), "opt": opt.state_dict(), "epoch": epoch},
                    os.path.join(d, CKPT_FILE),
                )
                checkpoint = ray.train.Checkpoint.from_directory(d)
            ray.train.report(metrics, checkpoint=checkpoint)

        # ---- 4. keep the resume path tested ----
        if rank == 0 and epoch == config["kill_at_epoch"]:
            print(f"[rank 0] --kill-at-epoch {epoch}: simulating a crash AFTER this "
                  f"epoch's checkpoint was reported. Rerun the same command to resume.")
            sys.stdout.flush()
            os._exit(1)


def build_dataset(root: Path, max_rows: int | None):
    ds = ray.data.read_parquet(str(root / "data"))
    if max_rows:
        ds = ds.limit(max_rows)
    return ds


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-name", required=True,
                   help="Unique per job. The SAME name + --storage-path resumes a previous run "
                        "(Ray Train V2 job-driver fault tolerance). A reused name silently resumes.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--storage-path", type=Path, default=Path.home() / "ray_train_runs")
    p.add_argument("--max-failures", type=int, default=0,
                   help="FailureConfig.max_failures: in-process worker retries. 0 = none (default), -1 = unlimited.")
    p.add_argument("--kill-at-epoch", type=int, default=-1,
                   help="Rank 0 os._exit(1)s after reporting this epoch's checkpoint. -1 = never.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="ds.limit(N) for fast iteration. Full dataset is 127,500 rows.")
    p.add_argument("--shuffle-buffer", type=int, default=0,
                   help="local_shuffle_buffer_size for iter_torch_batches. 0 = off. (Stretch goal)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.root / "data").exists():
        print(f"error: {args.root / 'data'} not found", file=sys.stderr)
        return 2

    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "kill_at_epoch": args.kill_at_epoch,
            "shuffle_buffer": args.shuffle_buffer,
        },
        scaling_config=ray.train.ScalingConfig(num_workers=args.num_workers, use_gpu=False),
        run_config=ray.train.RunConfig(
            storage_path=str(args.storage_path),
            name=args.run_name,
            failure_config=ray.train.FailureConfig(max_failures=args.max_failures),
        ),
        datasets={"train": build_dataset(args.root, args.max_rows)},
    )

    try:
        result = trainer.fit()
    except Exception as e:  # V2 raises on unrecovered failure; keep the CLI honest about it
        print(f"training failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if getattr(result, "error", None):
        print(f"training ended with error: {result.error}", file=sys.stderr)
        return 1

    print("result.metrics    =", result.metrics)
    print("result.checkpoint =", result.checkpoint.path if result.checkpoint else None)
    print(f"run dir           = {args.storage_path / args.run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())