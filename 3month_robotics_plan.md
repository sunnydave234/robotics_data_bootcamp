# 3-Month Robotics ML Learning Roadmap

---

## Month 1 · Project: `robot-data-forge`
### A Production Robot Learning Data Engine

You'll build the exact system Mundane, 1X, and Rhoda AI are all hiring for right now: a pipeline that turns raw robot demonstrations into versioned, queryable, training-ready datasets. By end of month you can load a real Open-X robot episode, watch it play back, query it by task type, and feed it into a PyTorch DataLoader — all with full DVC lineage. The model you train this month will be garbage. That's fine. The infrastructure won't be.

**Key Technologies:** Open-X Embodiment · HDF5 · DVC · PyTorch Dataset · episode ingestion · multimodal data

---

### Week-by-Week Build

#### Week 1 — Get a Robot Episode on Screen
*~20 hrs*

- Install PyTorch with MPS backend, verify it's using your M4 Max GPU: `torch.device("mps")`
- Download 20 episodes from Open-X Embodiment (`fractal20220817_data` on HuggingFace) — just load and inspect the raw structure
- Write a script that reads one episode and prints its schema: image shape, action dims, episode length, timestamp range
- Build a basic episode visualizer: loop through frames, overlay action vector as text, save as MP4 with OpenCV

**Deliverable:** A video file you can play that shows a robot arm doing something, with action data overlaid. Push it to GitHub.

**Concepts as needed:**
- RLDS / TFDS episode schema
- action space (read 1 section of Open-X paper §2)

> **DE Shortcut:** You already know how to inspect an unfamiliar data format — you've done this with FHIR, Delta Tables, EHR schemas. Same instinct. The HuggingFace datasets API is basically a managed S3 reader with a schema registry. It's slower than boto3 and less flexible. Treat it the same way.

---

#### Week 2 — Build the Ingestion Pipeline: Raw → Validated HDF5
*~20 hrs*

- Write `ingest.py`: converts Open-X episodes to a normalized HDF5 schema — `/observations/images`, `/observations/state`, `/actions` — with episode metadata as attributes
- Write `validate.py`: check every HDF5 file for NaNs, shape consistency, timestamp monotonicity, action/frame count parity — same data quality instinct as your FHIR pipeline
- Generate a `metadata.parquet` index: `episode_id`, `task_type`, `success`, `frame_count`, `file_path` — enables filter-without-opening

**Deliverable:** 20+ episodes ingested, validated, indexed. A `query.py` script that filters episodes by task type using the Parquet index and returns episode IDs in under 100ms.

**Concepts as needed:**
- h5py quick start (30 min, h5py.org)
- why chunked HDF5 is faster for random access

> **DE Shortcut:** Your `metadata.parquet` index is exactly the same pattern as your OpenSearch metadata layer at UHG — a lightweight index that avoids full dataset scans. The HDF5 files are your "raw EHR records." You've built this architecture before. Name it that way in your README.

---

> ### 🎯 Wow Moment 1 — End of Week 3
> **Your first robot episode plays back from your own pipeline**
>
> You load an episode from your HDF5 store, feed it through a PyTorch Dataset, and watch the robot's camera frames render in sequence with the predicted action vector plotted live. It's just numpy arrays and matplotlib — but it's real robot data that you ingested, normalized, and own. This is the moment it stops feeling like a tutorial.

---

#### Week 3 — PyTorch Dataset + DataLoader over Your HDF5 Data
*~20 hrs*

- Build `RobotEpisodeDataset(Dataset)`: reads from your HDF5, handles flat-index → `(episode, frame)` mapping, returns observation dict + action tensor
- Add normalization: compute action/state mean and std across your dataset, apply in `__getitem__`, save stats to JSON as a dataset artifact
- Benchmark DataLoader configs: `num_workers ∈ {0,2,4,8}` × `pin_memory ∈ {T,F}` — measure samples/sec. Write results to a markdown table.

**Deliverable:** A notebook that visualizes 5 consecutive frames from a loaded episode with action overlays. Numbers in your benchmark table. Something to show.

**Concepts as needed:**
- PyTorch Dataset `__getitem__` (one doc page, 20 min)
- why `pin_memory` matters (or doesn't on unified memory)

---

#### Week 4 — DVC Pipeline + End-to-End Reproducibility
*~20 hrs*

- Wire everything into a `dvc.yaml` pipeline: stage 1 = ingest, stage 2 = validate, stage 3 = build_index, stage 4 = compute_stats — each stage is a versioned artifact
- Write `reproduce.sh`: clone → `dvc repro` → identical output. Test it in a fresh directory.
- Train a trivial behavior cloning baseline (MLP: state → action) for 10 epochs just to verify the DataLoader works end-to-end. Loss should decrease. Model quality doesn't matter.
- Polish the README: architecture diagram, reproduce instructions, benchmark table, dataset stats table, link to your episode visualizer video

**Concepts as needed:**
- DVC "Data Pipelines" getting started (30 min)
- artifact lineage vs metric logging (one blog post)

> **DE Shortcut:** DVC pipelines are Airflow DAGs without the scheduler overhead. The `dvc.yaml` is your `dag.py`. The "remote" is your S3 bucket. You've thought about this problem at production scale already — the concepts will land in 30 minutes.

---

### ✅ End of Month 1 — What You Can Show

Open the terminal. Run `python query.py --task "pick and place" --min_success_rate 0.8` and watch it return a filtered list of episode IDs from your Parquet index in milliseconds. Open one episode, render it as a video. Run `dvc dag` and show the pipeline graph. Run `reproduce.sh` in a fresh clone and show it rebuilds everything. This is a 4-minute demo that shows a data engineer who understands the robot learning domain.

---
---

## Month 2 · Project: `robot-policy-lab`
### Train, Compare, and Analyze Two Manipulation Policies

You'll take the data engine you built in Month 1 and use it to actually train a robot manipulation policy. By end of month you'll have a W&B dashboard showing two policies (ACT and Diffusion Policy) competing on the PushT task — with your own hard-example mining loop improving both. The first time the policy successfully pushes the T-block in simulation is the moment this whole thing feels worth it.

**Key Technologies:** ACT policy · Diffusion Policy · LeRobot · W&B tracking · hard-example mining · simulation eval

---

### Week-by-Week Build

#### Week 1 — Get LeRobot Training Working Locally
*~20 hrs*

- Clone and install LeRobot: `pip install -e ".[pusht]"` — run their example training script unmodified first
- Get ACT training on PushT running on your Mac Studio MPS backend — you'll need to patch the device handling since it targets CUDA by default
- Read the training script source and annotate it: what is the training loop doing, where does data come from, what does the eval loop measure

**Deliverable:** A training run in progress with loss decreasing. Screenshot of the loss curve. One sentence in your README explaining what ACT is and why it works.

**Concepts as needed:**
- ACT paper §3 method section only (30 min)
- what behavior cloning is (one paragraph — you already understand it)

---

#### Week 2 — Wire Your Month 1 Data Pipeline into LeRobot
*~20 hrs*

- Write a `RobotForgeDataset` adapter that wraps your Month 1 HDF5 pipeline in LeRobot's expected interface — don't rewrite LeRobot, bridge it
- Add W&B integration to your training loop: log config, loss per step, eval reward per epoch, dataset artifact (DVC version pinned), GPU memory usage
- Implement full checkpointing: model + optimizer + epoch + best reward + rng state. Verify resume-from-checkpoint produces identical subsequent loss values.

**Deliverable:** A W&B run URL in your README. The run shows: which dataset version trained it (DVC hash), the eval curve, and the best checkpoint as a W&B artifact.

**Concepts as needed:**
- W&B Artifacts (one doc page, 20 min)
- what checkpoint resume requires (optimizer + rng state — the doc page explains it)

> **DE Shortcut:** W&B Artifacts are the same mental model as your DVC lineage tracking, but hosted. W&B run = your Airflow DAG run. Artifact = your versioned S3 object. The only new concept is that W&B tracks the model's performance alongside the artifact, not just the artifact itself.

---

> ### 🎯 Wow Moment 2 — End of Week 3
> **Your trained policy successfully pushes the T-block in simulation**
>
> The eval loop renders the PushT simulation and your policy — trained on data you ingested and versioned yourself — moves the robot arm to push the T-block into the goal zone. It won't work every time. It might work 20% of the time. But those 20% runs will feel like something. This is the point where "robot learning" stops being conceptual.

---

#### Week 3 — Train ACT, Evaluate in Simulation, Record Results
*~20 hrs*

- Run a full ACT training run to convergence (~100+ epochs on PushT). Watch the eval curve. Note when it plateaus.
- Render 10 evaluation rollouts as videos. Save them. You'll use these in your README and GitHub.
- Identify failure episodes: which initial conditions does the policy fail on? Build a simple failure classifier that flags these in your dataset index.

**Deliverable:** 10 eval rollout videos in your repo. A table: epoch vs success rate. A list of identified failure modes from your visual inspection.

**Concepts as needed:**
- success rate as the primary evaluation metric for manipulation policies

---

#### Week 4 — Add Diffusion Policy + Hard-Example Mining Loop
*~20 hrs*

- Train Diffusion Policy on the same dataset with identical training budget. The LeRobot framework supports it — swap the config. This gives you your comparison baseline.
- Implement hard-example mining: use your failure classifier from Week 3 to up-weight failure episodes in your `WeightedEpisodeSampler`. Re-train ACT with this sampler. Measure delta in success rate.
- Build a comparison dashboard in W&B: two policies, three training runs (ACT baseline, DP baseline, ACT + hard-example mining), all on the same axes.

**Deliverable:** A 1-page analysis in your README: which policy won, by how much, and whether hard-example mining helped. Real numbers, real opinion.

**Concepts as needed:**
- Diffusion Policy paper §1 intro + Fig 2 (20 min)
- curriculum learning in 1 paragraph (you don't need a paper — just understand the intuition)

> **DE Shortcut:** Hard-example mining is weighted sampling — something you've done with imbalanced datasets before. The robot version just uses task performance as the weight signal instead of class frequency. Your existing `WeightedEpisodeSampler` from Month 1 Week 4 is already the right structure. Add a column to your Parquet index: `failure_weight`. Done.

---
---

## Month 3 · Project: `edge-policy`
### Quantize, Deploy, and Build a Fleet Data Collector on RPi5

You'll take the policy you trained in Month 2, strip it down for edge deployment, and get it running as a real-time inference loop on a Raspberry Pi 5. Then you'll flip the RPi around: instead of running inference, it'll act as a robot data collection node — buffering sensor data and flushing to your Mac Studio when it hits a threshold. By the end you have a closed loop: RPi5 collects data → Mac Studio trains → RPi5 runs inference. The whole stack. On your own hardware.

**Key Technologies:** INT8 quantization · ONNX export · RPi5 deployment · CoreML/ANE · fleet data collection · edge inference

---

### Week-by-Week Build

#### Week 1 — Quantize Your Trained Policy + Benchmark on Mac Studio
*~20 hrs*

- Apply INT8 Post-Training Quantization (PTQ) to your best ACT checkpoint using PyTorch's quantization API
- Measure and record: model size (MB), inference latency p50/p95 (ms), success rate degradation on PushT eval — fill out a benchmark table
- Export to ONNX: `torch.onnx.export`, verify output parity with `onnxruntime`, measure ONNX Runtime latency vs PyTorch on CPU
- Convert to CoreML using `coremltools` and benchmark on your Mac Studio's Apple Neural Engine — this is unique to your hardware

**Deliverable:** A benchmark table in your README: 4 configurations (FP32 MPS / INT8 CPU / ONNX Runtime / CoreML ANE) × 3 metrics (size, latency, accuracy). Real numbers.

**Concepts as needed:**
- PTQ vs QAT (one PyTorch doc page, 20 min)
- what ONNX actually is (5-min explainer video is fine)

---

#### Week 2 — Deploy ONNX Model to RPi5, Measure Real Edge Latency
*~20 hrs*

- Set up RPi5: Python 3.11, ONNX Runtime ARM wheel, numpy, opencv — SSH access from Mac Studio for fast iteration
- Write an `infer.py` that runs the ONNX model in a loop, feeding dummy observations and measuring latency with `time.perf_counter()`
- Profile where time is spent on the RPi5: is the bottleneck preprocessing (image resize/normalize) or model inference? Fix the bottleneck.
- Target: <200ms per inference step on RPi5. Document what you had to sacrifice (input resolution, context length) to get there.

**Deliverable:** A screen recording of the RPi5 running inference with latency printed to stdout. Push the video to your repo. This is the demo moment.

**Concepts as needed:**
- ARM memory hierarchy basics (10 min — explains why ONNX Runtime is faster than PyTorch on ARM)

---

> ### 🎯 Wow Moment 3 — Week 2 of Month 3
> **Policy inference running on your Raspberry Pi 5 at sub-200ms**
>
> You SSH into the RPi5, run `python infer.py`, and watch inference timestamps scroll by — a robot policy you trained on your own data, running in real time on $80 hardware. This is what "deploying models to edge devices with real-time latency constraints" means. You've now done it. Not in theory.

---

#### Week 3 — Turn RPi5 into a Fleet Data Collection Node
*~20 hrs*

- Write a `data_collector.py` daemon on the RPi5 that logs "sensor readings" (timestamp + simulated observation dict + action) to a local ring buffer
- Implement the "intelligent trigger" pattern: flush to Mac Studio over SSH/SCP only when buffer exceeds a threshold OR a time window expires — not after every sample
- On the Mac Studio side, write a `fleet_receiver.py` that accepts flushes, validates them (same validator from Month 1), and appends to your HDF5 store with auto-generated episode IDs

**Deliverable:** The RPi5 collecting data at 10Hz and flushing to Mac Studio every 30 seconds. Your HDF5 store grows. Your Parquet index updates. Your Month 1 pipeline processes the new episodes.

> **DE Shortcut:** This is an SQS producer/consumer pattern you've built before. RPi5 is the producer with a local queue (ring buffer). Mac Studio is the consumer. The "intelligent trigger" is just a batch flush condition — exactly what your Lambda functions do when they drain SQS queues at UHG. You don't need to learn this. You need to implement it in a robotics context.

---

#### Week 4 — Close the Loop + Production-Polish Everything
*~20 hrs*

- Add Prometheus metrics to your fleet receiver: `episodes_received_total`, `bytes_ingested_total`, `validation_failures_total` — scrape from a local Prometheus instance, build a Grafana dashboard
- Write a `retrain_trigger.py` that watches your HDF5 store and fires a training run (using your Month 2 setup) when N new episodes have arrived from the fleet
- Polish all three repos: consistent READMEs, architecture diagrams, reproduce instructions, benchmark tables, W&B run URLs, demo videos
- Write two technical blog posts (500 words each): one on your data engine architecture, one on your quantization benchmark results — publish on Substack or Medium

**Deliverable:** The full closed-loop demo (below).

**Concepts as needed:**
- Prometheus data model (one doc page, 15 min — you know observability already)

> **DE Shortcut:** You already use Terraform and manage 20+ AWS services. Standing up a local Prometheus + Grafana stack in Docker Compose is trivial by comparison — 45 minutes including the dashboard. The only new thing is the Prometheus data model (counters vs gauges vs histograms). Read the one-page explainer and you're done.

---

### ✅ End of Month 3 — The Full Closed-Loop Demo

**Screen 1:** RPi5 terminal running `data_collector.py` — timestamps scrolling at 10Hz.  
**Screen 2:** Mac Studio Grafana dashboard showing `episodes_received_total` ticking up every 30 seconds.  
**Screen 3:** W&B showing a retraining run firing automatically when 50 new episodes arrive.  
**Screen 4:** The RPi5 running `infer.py` with sub-200ms latency.

This is a live demo of the full robot learning stack — data collection, ingestion, training, deployment — running on hardware you own. No cloud, no simulation, no toy dataset.
