import ray, ray.train, ray.train.torch

def fn(config):
    shard = ray.train.get_dataset_shard("train")
    ids = []
    for batch in shard.iter_batches(batch_size=4):
        ids.extend(batch["id"].tolist())
    rank = ray.train.get_context().get_world_rank()
    print(f"rank={rank} n_rows={len(ids)}   ids={sorted(ids)}")
    ray.train.report({"n_rows": len(ids)})

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    trainer = ray.train.torch.TorchTrainer(
        fn,
        scaling_config=ray.train.ScalingConfig(num_workers=2, use_gpu=False),
        datasets={"train": ray.data.range(21)},
    )
    trainer.fit()