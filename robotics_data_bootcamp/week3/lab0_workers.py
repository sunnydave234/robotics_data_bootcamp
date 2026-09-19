# lab0_workers.py
import os
import ray, ray.train, ray.train.torch

def fn(config):
    ctx = ray.train.get_context()
    print(f"rank={ctx.get_world_rank()}  world={ctx.get_world_size()}  "
          f"pid={os.getpid()}  ppid={os.getppid()}  device={ray.train.torch.get_device()}")
    ray.train.report({"reported_by_rank": ctx.get_world_rank()})

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    print(f"driver pid={os.getpid()}")
    trainer = ray.train.torch.TorchTrainer(
        fn, scaling_config=ray.train.ScalingConfig(num_workers=2, use_gpu=True),
    )
    print("result.metrics =", trainer.fit().metrics)