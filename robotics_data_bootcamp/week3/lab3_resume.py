# lab3_resume.py
import os, tempfile
from pathlib import Path
import torch, torch.nn as nn
import ray, ray.train, ray.train.torch

STORAGE = str(Path.home() / "ray_train_runs")

def fn(config):
    model = nn.Linear(14, 14)
    start_epoch, opt_state = 0, None
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            state = torch.load(os.path.join(d, "state.pt"))
            model.load_state_dict(state["model"])      # load BEFORE prepare_model -- docs' order
            opt_state, start_epoch = state["opt"], state["epoch"] + 1
        print(f"RESUMED at epoch {start_epoch}")
    model = ray.train.torch.prepare_model(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # build AFTER prepare_model -- see Common Mistakes
    if opt_state:
        opt.load_state_dict(opt_state)

    x = torch.randn(256, 14); y = x + 0.1 * torch.randn(256, 14)   # toy data: y ~ x, like state->action
    for epoch in range(start_epoch, config["epochs"]):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        with tempfile.TemporaryDirectory() as d:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
                       os.path.join(d, "state.pt"))
            ray.train.report({"epoch": epoch, "loss": loss.item()},
                             checkpoint=ray.train.Checkpoint.from_directory(d))
        print(f"epoch={epoch} loss={loss.item():.5f}")
        if epoch == config.get("kill_at_epoch", -1):
            print("simulating a crash AFTER this epoch's checkpoint was reported")
            os._exit(1)

if __name__ == "__main__":
    import sys
    run_name = sys.argv[1]; kill_at = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    trainer = ray.train.torch.TorchTrainer(
        fn, train_loop_config={"epochs": 6, "kill_at_epoch": kill_at},
        scaling_config=ray.train.ScalingConfig(num_workers=1, use_gpu=False),
        run_config=ray.train.RunConfig(storage_path=STORAGE, name=run_name),
    )
    print(trainer.fit().metrics)