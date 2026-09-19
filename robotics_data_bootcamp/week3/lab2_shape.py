from pathlib import Path
import ray, ray.train, ray.train.torch
import numpy as np, torch
from tensor_utils import to_2d_float_array
import torch.nn as nn

STATE_DIM = ACTION_DIM = 14
DATA_DIR = Path.home() / ".cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet/data"

def collate(batch: dict) -> dict:
    # batch is numpy dict-of-arrays you know from Day 1 -- smae shape gotcha applies.
    return {
        "state":    torch.as_tensor(to_2d_float_array(batch, "observation.state", STATE_DIM)),
        "action":    torch.as_tensor(to_2d_float_array(batch, "action", ACTION_DIM)),
    }

## Step 2 `fn` with `collate_fn`
# def fn(config):
#     shard = ray.train.get_dataset_shard("train")
#     # batch = next(iter(shard.iter_torch_batches(batch_size=32)))
#     batch = next(iter(shard.iter_torch_batches(batch_size=32, collate_fn=collate)))
#     for k, v in batch.items():
#         print(f"{k:24s} {type(v).__name__:8s}   shape={tuple(getattr(v, 'shape', ()))}  dtype={getattr(v, 'dtype', None)}")
#     ray.train.report({})

def fn(config):
    model = ray.train.torch.prepare_model(nn.Linear(STATE_DIM, ACTION_DIM))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    shard = ray.train.get_dataset_shard("train")
    first = last = None
    for batch in shard.iter_torch_batches(batch_size=256, collate_fn=collate):
        loss = nn.functional.mse_loss(model(batch["state"]), batch["action"])
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    ray.train.report({"first_loss": first, "last_loss": last})

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    ray.train.torch.TorchTrainer(
        fn,
        scaling_config=ray.train.ScalingConfig(num_workers=1, use_gpu=False),
        datasets={"train": ray.data.read_parquet(DATA_DIR)}
    ).fit()