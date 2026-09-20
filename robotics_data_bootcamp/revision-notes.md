# Robotics Data Engineering — Revision Notes

Cumulative revision notes, Weeks 1–3. One section per day: the core problem in plain language, how it's solved technically, real numbers where confirmed, the key gotcha(s), and an interview framing. Built from the daily notebook, `progress-log.md`, and this project's working code — not a re-read of the full session transcripts.

A few entries are flagged **[reconstructed]** where the source material was thin — treat those as a best-effort summary, not a verified record.

---

## Week 1 — LeRobot Dataset Fundamentals

### Day 1 — Episode & Dataset Structure

**The core idea.** A LeRobot dataset isn't one flat file — it's three layers: dataset-level metadata (`ds.meta`), episodes (one demonstration each), and frames (one timestep each, with images/state/action). Two access paths exist for a reason: `ds.hf_dataset[i]` reads tabular data only (Arrow/Parquet, no video decode), while `ds[i]` decodes every camera's video for that frame. Reaching for `ds[i]` when you only need a timestamp is the single most common way to make a script slow for no reason.

```
ON DISK                                    IN PYTHON
meta/info.json, stats.json, episodes/*  →  ds.meta (fps, camera_keys, features, episodes table)
data/*.parquet, videos/*.mp4            →  ds.hf_dataset (tabular, no decode) / ds[i] (decodes video)
```

**Built:** `episode_card.py` — CLI describing a single episode.

**Key gotcha.** `LeRobotDataset(repo_id, episodes=[N])` **re-indexes the dataset locally** to `0..length-1`. Calling `ds[N]` afterward doesn't mean "give me episode N" — it means "give me frame N *within the filtered subset*." Silent for fixed-shape fields like action/state (always shape 14 regardless), but produces an `IndexError` on a short or late episode. This is the same bug pattern Day 4 later names explicitly — already present here, just not yet caught.

**Interview framing:** "I found a silent-failure class where filtering a dataset changes what a positional index *means*, not just what it *contains* — the kind of bug that doesn't crash, so it survives code review and shows up as a training-data corruption instead."

---

### Day 2 — Sensors and Modalities

**The core idea.** Two headaches are structural to physical AI, not specific to any one dataset:

1. **Asynchronous frequencies.** Sensors run on different physical clocks — a camera takes real time to process a pixel array (~15Hz), while a joint encoder is just an electrical signal (50–500Hz). At any instant you have fresh joint data next to stale camera data. In data-engineering terms, every sensor firing becomes one row, producing a sparse wide table with gaps — handled by forward-filling the last known value.
2. **Sensor blindness.** A policy trained on Robot A (with a force sensor) breaks on Robot B (without one) — the pipeline expects a `force_torque` column that doesn't exist. The fix is structural: check which sensors are actually present before reading them, rather than assuming a fixed schema.

A third, subtler case: a sensor can be *present but broken* — a dead depth camera outputs an array of all zeros, which looks structurally valid (the key exists) but is garbage. Detect it with variance: if `arr.max() == arr.min()` across samples, the sensor is flatlined, not just quiet.

**Built:** `sensor_inventory.py` (coverage matrix across real episodes) + an fps-verification helper (`measure_fps`: sample timestamps across an episode, take `1/median(dt)`, compare against `meta.fps`).

**Interview framing:** "Two failure modes matter in multi-sensor robotics pipelines: a sensor that's absent (schema mismatch, loud) and a sensor that's present but dead (silent, needs a variance check to catch). Building for the first without the second still ships broken data."

---

### Day 3 — Timestamps, Frequency, and Synchronization

**The core problem.** Camera and joints record at different native rates. `LeRobotDataset` collapses both onto one timeline — each `ds[i]` looks like one clean row (one timestamp, one state, one action, one image) — but that image was pulled by *seeking* into the camera's MP4 at that timestamp, a black box you never see directly, you only inherit the result.

**Three failure modes, and why they're dangerous:**

- **Drift** — camera and joint clocks slowly disagree; the gap grows over the episode.
- **Dropped frames** — a frame should have been captured and wasn't; a hole in the sequence.
- **Duplicate/held frames** — no new frame was ready, so the pipeline reused the last one; two rows point at the same image.

None of these crash. None show up as a training-loss spike. They silently mispair an image with the wrong action — the model still trains, the loss still goes down, and the policy fails only later, on real hardware. Same failure *shape* as Day 1's silent indexing bug, one layer higher in the stack.

**The key distinction — direct vs. proxy detection:**

- Drift and dropped frames are detectable **directly** from timestamp math alone (gaps vs. the nominal `1/fps` interval) — cheap.
- Duplicate/held frames are only detectable **by proxy** — timestamps look perfectly normal; you have to decode frames and diff pixels — expensive.

**Detection cheatsheet:**

- **Extract cheaply:** never call `ds[i]` just to read a timestamp — it decodes every camera needlessly. Use `ds.hf_dataset[from_idx:to_idx]["timestamp"]` instead: tabular only, no video touch.
- **FPS sanity check:** `1/median(deltas) ≈ meta.fps`. Use **median**, not mean — median is rank-based (an outlier just sits at the edge of the sorted list, ignored); mean is magnitude-based (any outlier drags it, worse the bigger it is). Two separate jobs: build the trustworthy "what's normal" yardstick first (median), then hunt for anomalies by comparing the raw deltas — especially the max — against that yardstick. A max/median ratio near 1.0 is boring; 2×, 10×, 100× means something real happened.
- **Dropped frames:** `deltas = np.diff(timestamps)`; flag where `deltas > tolerance * expected_dt`. The tolerance (~1.5×) exists because real capture always has tiny natural jitter — too tight and you flag normal noise as drops, too loose and you miss real ones.
- **Duplicate `frame_index`:** a bookkeeping bug, not a camera issue. `np.unique(frame_index_col, return_counts=True)`, flag any count > 1.
- **Held frames:** stream one frame at a time (peak memory = 2 frames, not a whole episode), `np.allclose(curr, prev)` between consecutive frames, and build run-lengths of consecutive matches. Runs clustering tightly around one small integer (e.g., always ~2) means expected resampling — a slow camera stretched onto a faster clock, harmless. Wandering run lengths mean something is genuinely inconsistent.
- **Drift** needs *separate* per-modality timestamps — rare in this dataset (one shared `timestamp` column), guaranteed in Week 2's MCAP data.
- **Cost-tiered strategy:** cheap tabular checks on 100% of episodes → use those flags to prioritize which episodes get the expensive visual pass, plus a few random ones (held frames can hide in a tabular-clean episode) → always distinguish "checked and clean" from "never checked."
- **Always validate a detector against a synthetic array with a known injected fault** — the only way to know it actually catches what you think it catches, instead of just running clean on real data by luck.

**Built:** `sync_checker.py` (tiered tabular + visual checks, all detectors synthetic-fault tested).

**Key gotcha.** Naive per-frame extraction (`[ds[i] for i in range(...)]`) decodes every camera for every frame just to read a timestamp scalar — OOM'd Colab. Fixed via `ds.hf_dataset`. Second finding: this dataset exposes one shared `timestamp` per row, not separate per-camera clocks, so drift is only observable *indirectly*, via held-frame runs.

**Interview framing:** "I built a cost-tiered sync-quality pipeline — cheap timestamp checks on every episode, expensive pixel-diff checks only where flagged plus random sampling, because held frames can hide in timestamp-clean episodes. Every detector is validated against a synthetic fault before I trust it on real data, because this whole bug class is invisible in training loss."

---

### Day 4 — Trajectories, Actions, and Coordinate Frames

**The core idea.** `action` and `observation.state` are both `float32[14]` and describe the *same* 14 joint names. The tempting wrong assumption: `predicted_next = state[:-1] + action[:-1]`, treating `action` as a delta ("move 5 degrees from here"). Wrong for this dataset. ALOHA uses **leader-follower teleoperation**: a human physically moves a leader arm, and the follower arm mechanically copies its exact position, joint for joint — a puppet mirroring its controller. `action` and `state` share identical joint names because they're the *same kind of thing* — one is "the target it was told to reach," the other is "where it actually was." The correct read: `action[t]` is an **absolute** commanded joint target, like GPS coordinates for a destination, not a delta instruction.

**How to check tracking quality:** `action_following_error = ||state[1:] - action[:-1]||` — how far the follower ended up from what it was told a step earlier. To flag genuinely bad episodes rather than just "the worst one," compute each episode's mean error, then a **z-score** against the median and std *across* episodes — flag `z > 2.0` as statistical outliers, not just nonzero.

**Debugging challenge:** loading just one episode via `LeRobotDataset(repo_id, episodes=[60])` changes its frame indexing to `[0, len)` — the same Day 1 bug, now hitting a *different* function. **The fix:** filter `ds.hf_dataset.to_pandas()` by the `episode_index` **column value**, sorted by `frame_index` — never by position against a filtered object. Formally: `dataset_from_index`/`dataset_to_index` are GLOBAL row numbers, but episode-filtering re-indexes `hf_dataset` LOCALLY — positional slicing with global indices against a filtered dataset silently returns an empty result. Confirmed against real, currently-open LeRobot GitHub issues (#816, #1783, #1895, #2678) — a live bug in the library, not a hypothetical.

**Built:** `trajectory_checker.py` (RAM-safe trajectory extraction, action-following anomaly detector, arm-flip detector, `--plot` mode that iterates `range(len(motor_names))` instead of a hardcoded 14).

**Interview framing:** "I diagnosed that this dataset's `action` field is an absolute joint target from leader-follower teleoperation, not a delta — get that wrong and every downstream policy trains on an inverted assumption. I also found and cited the exact open GitHub issues describing a global/local indexing bug I'd independently hit myself."

---

### Day 5 (Capstone) — Full Pipeline + Health Report

**The core idea.** Combine every prior check (schema/coverage, sync, action-following) into one per-episode health score — and the scoring *architecture* matters as much as the checks themselves.

**Built:** `week1_report.py` — full pipeline merging `describe_episode` + `sensor_coverage_matrix` + sync detectors + `action_following_error` by `episode_index`; a **gated** health scorer; markdown report with a CI exit code.

**Key gotcha.** `episode_card.py` never had `describe_episode()` factored out as a standalone function on Day 1 — everything was inline in `main()`. Had to refactor before the capstone's "import and call" pattern worked at all. Lesson: refactor for reuse *before* you need it, not after — Day 1's shortcut cost real rework time here.

**Second, more important gotcha.** A naive equal-weighted health score let a **schema-broken** episode score 0.69 — looks fine-ish, dangerously misleading. Averaging a catastrophic failure in with three healthy scores dilutes it into invisibility. Fixed with a **hard gate**: schema failures force the score to `0.0` and get marked "excluded," never folded into a weighted average with the other checks.

**Interview framing:** "A structural failure and a borderline quality issue aren't the same kind of bad, and averaging them together produces a misleadingly OK-looking score. I built a hard-gate architecture where catastrophic failures short-circuit to zero and get excluded from scoring entirely, rather than diluted into an average."


---

## Week 2 — Format Interop: MCAP, rosbag2, RLDS, HDF5

### Day 1 — LeRobot → MCAP Conversion

**The core idea.** MCAP is a self-describing, timestamped binary log format used in real robotics (via ROS 2 / rosbag2), built around one global wall-clock timeline — `int64` nanoseconds since epoch. LeRobot's own timestamp is the opposite: `float32` seconds **from episode start**, resetting to `0.0` every episode. Converting between the two means translating between two genuinely different notions of "time," not just relabeling a column.

**Key gotcha (the real one).** Writing `int(timestamp * 1e9)` straight through for a multi-episode file is structurally valid — the file opens without error — and silently wrong: since every episode's LeRobot timestamp resets to 0, episodes **interleave** on the global MCAP timeline instead of concatenating. The reported duration comes out as just the longest single episode, and the implied message rate comes out as `fps × n_episodes`. Fixed with a cumulative per-episode offset (`episode_time_offset_ns`, derived from each episode's declared length/fps) added before conversion.

**Second finding, numbers-backed.** Float32 timestamps mean the derived nanosecond gaps are *never* exactly identical — on a clean 1500-frame, 50Hz episode, only 2 of 1499 gaps land exactly on 20,000,000 ns; the rest spread between 19,998,550 and 20,000,458 ns. A concrete confirmation of "never `==` on timestamps," now with an actual number behind it.

**Third finding.** MCAP's `chunk_size` changes chunk count and seek performance only — never message content. A storage decision, not a semantic one — same lesson HDF5 chunking teaches later this week.

**Built:** `mcap_from_lerobot.py` (episode range resolved by `episode_index` *value*, not position — directly reusing Week 1's lesson; `--split-topics`, `--time-base episode|global`, `--compression`, `--chunk-size`) + `mcap_inspect.py` (cheap summary pass + full per-message pass; per-topic count/duration/implied-Hz/gap flags; non-zero exit for CI).

**Interview framing:** "Converting between data formats requires understanding what time actually *means* in each one — two 'timestamp' columns that look compatible can encode fundamentally different clocks, and naive conversion silently corrupts episode boundaries without ever throwing an error."

---

### Day 2 — rosbag2 and Custom Message Types

**The core idea.** MCAP can be rosbag2's **storage backend**, but a raw MCAP file isn't automatically a valid rosbag2 bag. It needs a companion `metadata.yaml` (storage format, ROS 2 profile, serialization format), and every message schema inside must use ROS 2's own CDR serialization with a matching type definition — not arbitrary JSON.

**The four-round debugging story** (a genuinely good one to have ready for an interview): the original plan — wrap a hand-rolled `metadata.yaml` around a custom-JSON-schema MCAP file, read it with `AnyReader` — failed with four validation errors that turned out to be **one architectural mismatch expressed four ways**, not four unrelated bugs: `serialization_format` had to literally be `"cdr"`; the profile had to be `"ros2"`; and — the one that couldn't be patched around — every schema's *encoding* had to be `ros2msg`/`ros2idl`, checked while cataloging schemas at file-open time, before a single message is read. Two of the four could be faked with a "deliberate lie" in the manifest; the fourth reads a real field from the real file, so it can't be. rosbag2's own docs confirm the actual scope: MCAP-as-backend is for *custom ROS 2 message types with real CDR encoding*, never arbitrary schemas smuggled in as JSON.

**The real fix:** define an actual ROS 2 message type via `rosbags`' `get_types_from_msg`, register it in a typestore, and write a genuine bag with `rosbags.rosbag2.Writer(storage_plugin=MCAP)` — letting the library produce a correct manifest instead of hand-guessing one.

**Gotchas found while rebuilding:** array fields need **numpy** arrays, not plain Python lists (`serialize_cdr` fails on lists with `AttributeError: 'list' object has no attribute 'view'`). And a custom type only exists in the process that registered it — `AnyReader` does not auto-discover a bag's embedded custom definitions; you must pass a typestore with the type pre-registered.

**Built:** `rosbag2_inspect.py` (reports an unreadable bag as a clean non-zero exit instead of crashing).

**Interview framing:** "I hit four validation errors in a row and, rather than patching each individually, recognized they were symptoms of one wrong architectural assumption — confirmed that against the library's own documented scope, and rebuilt around the legitimate path instead of continuing to patch a broken foundation."

---

### Day 3 — RLDS / Open X-Embodiment Schema Comparison

**The core idea.** Different labs publish robotics datasets in different formats even for similar tasks. Open X-Embodiment (OXE) is published as RLDS, a TensorFlow-native format. Diffing an OXE sub-dataset's schema against your own LeRobot-format dataset tells you what survives a format conversion, what changes, and — critically — what gets **dropped**.

**Approach-change gotcha.** The planned load path (a community HF mirror, `jxu124/OpenX-Embodiment`) is confirmed dead — Hugging Face's `datasets` 4.0.0 (July 2025) permanently removed script-based dataset loading, and that mirror was script-based. Rebuilt around `tfds.builder_from_directory` reading directly from the public `gs://gresearch/robotics` GCS bucket — confirmed against the official OXE repo, not a blog post.

**Two more gotchas:** a hardcoded "0.1.0 by default" version rule (from a 2023 blog post about the original release) doesn't hold for `aloha_mobile` (a 2024 addition) — real version is `0.0.1`, confirmed directly against the bucket. And naive "take the alphabetically-last directory" version discovery picked a GCS **placeholder marker** over the real directory, because the placeholder string sorts after it — fixed by filtering on the trailing `/` real subdirectories carry.

**RAM gotcha:** reading even one `aloha_mobile` episode crashed Colab. TFDS decodes every `Image` feature by default regardless of use. Fixed with `tfds.decode.PartialDecoding` to skip unused camera fields, plus TF's own documented `tf.data` RAM-safety options.

**Real, numbers-backed finding:** `aloha_mobile`'s RLDS `action` is `float32[16]`, vs. `lerobot/aloha_mobile_cabinet`'s `float32[14]` — the first concrete evidence (not just a course claim) that a `base_action` field was dropped in the LeRobot conversion. `observation.state` matches at `float32[14]` and camera names match exactly, supporting shared lineage despite the action-shape gap. Episode counts don't match (276 vs. 85) because `aloha_mobile` aggregates multiple tasks, not just cabinet — compare by schema/content, never by raw episode count.

**Built:** `rlds_schema_diff.py` (key-set diffs, shape/dtype mismatches, per-side episode totals) + `rlds_inspect.py` (single-dataset profiler).

**Interview framing:** "When a community dataset mirror silently breaks due to an upstream library change, go to the org's own bucket or official docs rather than patching around the mirror. I also found a real, numbers-backed schema gap — 16 vs. 14 action dims — across a real format conversion, exactly the due-diligence check that matters before trusting converted training data."

---

### Day 4 — Format Migration & Verification (v2.1 → v3.0)

**The core idea.** When a dataset *format spec itself* changes version, you need to verify a conversion didn't silently corrupt or drop data — element-wise comparison, not just "the script exited 0."

**Built:** `verify_conversion.py` (episode-index-*value*-matched, per-field element-wise diff, tolerance-gated for float precision, non-zero exit on drift) + `format_migration_gate.py` (generalized to Hub revisions or local DVC-style roots).

**Verified against real sources, not memory:** LeRobot GitHub issue #2689, confirmed open — a real, currently-unresolved report of erratic joint actions after this exact kind of conversion. Real `meta/info.json` pulled directly for `aloha_mobile_cabinet` (v3.0, 85 episodes, 127,500 frames, fps 50, AV1 codec, `chunks_size` 1000). The `revision="v2.1"` Hub tag pattern and the converter's write-access requirement both confirmed against separate real issues (#1998, #2446).

**Interview framing:** "Format migrations in ML data pipelines are a live source of silent corruption — I cited an actual open issue describing exactly this failure mode in a widely-used library, and built a verification tool that catches it by comparing values, matched by ID, not by trusting the migration script's exit code."

---

### Day 5 — HDF5 Capture-Layer Inspection **[reconstructed]**

*The session log for this day was thin — this section is rebuilt from what survived plus general knowledge of the format, not a full write-up. Flag anything below that doesn't match what you actually remember.*

**The core idea.** HDF5 is often the **raw capture-layer** format written directly by robot hardware during teleoperation — before any conversion to a training-friendly layout like LeRobot's Parquet+video. It supports per-dataset chunking and compression: the same "storage decision, not a semantic one" lesson as MCAP's `chunk_size` from Day 1, one layer earlier in the pipeline.

**The finding:** raw HDF5 capture includes fields that don't survive the LeRobot conversion — specifically `qvel` (joint velocities) and `base_action` (the mobile base's own action, separate from the arm). LeRobot's conversion only carries `action`, `state` (from `qpos`), and `effort` forward. Practical consequence: if you ever need to reconstruct a capture-layer file *starting from* a LeRobot dataset, `qvel`/`base_action` are structurally unrecoverable — that data is already gone by the time you have the LeRobot version.

**Worth noticing:** this independently corroborates Day 3's finding from a completely different angle — `aloha_mobile`'s RLDS action being `float32[16]` vs. LeRobot's `float32[14]` is consistent with `base_action` (plausibly a 2D mobile-base command) being exactly what's missing. Two unrelated checks, two days apart, pointing at the same gap.

**Built:** `inspect_hdf5.py` — walks an ALOHA-native HDF5 file and reports shape/dtype/chunk-shape/compression per dataset, and flags whether `effort`/`base_action` are present.

**Interview framing:** "Raw hardware capture often carries more signal than survives a training-oriented conversion. I traced a specific example — `qvel` and `base_action` — from raw HDF5 through to the Parquet layer and confirmed it doesn't survive, which independently corroborated a schema gap I'd found two days earlier from a completely different angle."


---

## Week 3 — Ray Data for Robotics Video

### Day 1 — Ray Data Fundamentals & the Tensor Shape Gotcha

**The core idea.** Ray Data represents a batch as a **dict of arrays**, not a list of row-dicts, and its execution is lazy — nothing runs until something (`.count()`, `.take()`) actually forces it. That changes what's "free" to check versus what silently triggers a full pass.

**Course-material correction, verified against live docs (Ray 2.56.0):** `ds.count()` on a bare `read_parquet()` is *not* a full scan — Ray reads Parquet footer metadata, the same trick Spark and DuckDB use. Still costs more than `LeRobotDatasetMetadata.total_frames` (scales with file count vs. zero file opens), but the original course assumption that `count()` would be slow was simply wrong. This fast path **breaks the instant any `map_batches` is chained** before it.

**The real gotcha, found via an actual debug print, not a guess.** Ray Data represented this dataset's `action` column (fixed-shape `float32[14]` per row) as a 1-D `dtype=object` array of per-row arrays — not the `(batch_size, 14)` 2D tensor Ray's own docs describe for fixed-shape columns. `np.linalg.norm(action, axis=-1)` on that shape silently reduced the *wrong* axis instead of erroring, producing a wrong-length output that looked like an off-by-something indexing bug, not a dtype issue.

**Two wrong hypotheses tried first — worth remembering as a debugging pattern:** (1) a `LimitOperator`/`take_batch()` interaction — disproven when `.materialize()` before `.take_batch()` changed nothing; (2) an in-place batch-mutation theory — disproven when switching to `{**batch, ...}` changed nothing either. **The same error surviving a fix is the signal the theory is wrong, not that the fix needs more tweaking.**

**Real fix:** `to_2d_float_array()` in `tensor_utils.py` — explicitly `np.stack()`s the object array before any math touches it.

**Real numbers (file-group partitioning, sourced from Anyscale's own published reference):** 85 episodes → 4 partitions on `aloha_mobile_cabinet`, because the 3 cameras roll to their 2nd video file at three *different* episodes (66, 79, 82) — proving a single-camera key alone gives a falsely-optimistic grouping. ~21× fewer video opens than naive per-episode parallelism (255 → 12) — smaller than DROID's published 135×, honestly explained by dataset size, not a failed replication.

**Built:** `tensor_utils.py`, `lab1_ray_data_intro.py`, `lab2_action_magnitude_flag.py` (a deliberately naive per-block flag, kept as a teaching artifact), `ray_bronze_ingest.py` (dataset-wide mean/std via one `ds.aggregate()` call — the real fix for Lab 2's per-block flaw, which only ever saw outliers relative to its own small batch).

**Interview framing:** "A library's own docs described a behavior that didn't hold for this specific dataset — I found the real shape via a debug print, not by trusting the docs, and built a small tested utility every downstream script now imports. I also caught two wrong hypotheses by recognizing that an error surviving a 'fix' means the theory is wrong, not the patch."

---

### Day 2 — File-Group Partitioning, and the decord → torchcodec → PyAV Journey

**The core idea.** LeRobot v3.0 packs *multiple* episodes into one physical video file per camera, not one file per episode. Naive per-episode reads re-open and re-seek into the same physical file repeatedly. File-group partitioning groups episodes sharing the exact same `(chunk_index, file_index)` across *every* camera into one task, so each physical file is opened exactly once.

**Real numbers, confirmed by an actual run:** 85 episodes → 4 partitions; 255 → 12 video opens (21.2×); all three episodes in the smallest partition came back `expected == found` exactly — no off-by-one at the `[lo, hi)` timestamp boundary.

**The debugging/pivot journey — a good one to have ready for an interview:**

1. **Plan:** use `ray.data.read_videos()`, Ray Data's built-in video reader.
2. **Blocked:** it depends internally on `decord` (confirmed via the real traceback). decord's compiled binary needed an old FFmpeg library version that a Homebrew upgrade had removed — and decord is confirmed essentially unmaintained upstream.
3. **Fix #1:** swapped to `torchcodec` — already installed as LeRobot's own default decode backend — worked, confirmed correct end to end on a real run.
4. **Fix #2 (final):** before calling it done, checked what real practitioners actually do. Ray's own Data maintainer publicly recommends a **custom Datasource** over the built-in reader for open-source Ray; Anyscale's own published reference pipeline for this exact job uses **PyAV**, not decord or torchcodec. Rewrote to match: open the file once, seek once to the first episode's `from_timestamp`, decode forward continuously across the whole group.

**A satisfying, independently-earned validation:** the partitioning logic (composite key + contiguous-merge + assertion) turned out structurally near-identical to Anyscale's own published function — derived independently, before ever reading theirs.

**Key technical lessons:** `meta.episodes` is Arrow-backed — always `.to_pandas()` first. Column names containing `/` silently break pandas' `itertuples()` (renamed to generic `_1`/`_2` fields) — use `.to_dict("records")`. Lambda **default arguments** (`lambda r, lo=lo: ...`) prevent a real "late binding" bug when a lambda is built fresh inside a loop. `videos/{cam}/from_timestamp`/`to_timestamp` place an episode *inside* its shared video file (seconds, file-relative) — a different concept from `dataset_from_index`/`to_index` (global row numbers). PyAV's `seek()` lands on the nearest keyframe *at or before* the target, never the target itself — you must filter `[lo, hi)` after decoding forward.

**Interview framing:** "I had a working, verified fix, but before treating it as done, I checked what the actual industry reference implementation for this exact problem does — found it uses a different library for defensible reasons — and rewrote to match. That's the gap between 'it works' and 'it works the way production systems actually do it.'"

---

### Day 3 — Ray Train: Four Mechanisms, Quick-Lookup Format

*Format note: starting this day, each mechanism is paired with the exact code that implements it in `ray_train_stub.py`, instead of a prose paragraph — so a glance answers "which point is solved by what code" without re-reading the file. All four mechanisms below were run for real this session (Labs 0–3), not just reasoned from docs — see the "confirmed by a real run" lines for exactly what was and wasn't personally verified.*

**The problem these four solve, in one line:** training across several GPUs means splitting the data fairly, keeping every GPU's copy of the model identical, saving progress so a crash doesn't cost everything, and being able to pick back up — Ray Train's `TorchTrainer` owns all four so you don't hand-write them.

---

**1. Sharding — split the dataset across workers**

`ray_train_stub.py → train_loop_per_worker()`:
```python
shard = ray.train.get_dataset_shard("train")
...
for batch in shard.iter_torch_batches(
    batch_size=config["batch_size"],
    collate_fn=collate_fn,
    local_shuffle_buffer_size=config["shuffle_buffer"] or None,
):
```
Confirmed by a real run (Lab 1, `ray.data.range(21)`, `num_workers=2`): 10 rows and 10 rows — `equal=True` drops the 21st row (`id=20`) to keep both shards exactly balanced. The split was **interleaved** (rank 0 got every even id, rank 1 every odd), not contiguous blocks — not a documented guarantee, just how this run's blocks happened to land. Never build logic that depends on *which* rows land where, only that the split is disjoint and equal-sized.

---

**2. DDP — keep every worker's model copy identical**

`ray_train_stub.py → train_loop_per_worker()` + `unwrap()`:
```python
model = ray.train.torch.prepare_model(model)                 # DDP-wraps ONLY if world_size > 1
opt = torch.optim.Adam(model.parameters(), lr=config["lr"])  # built AFTER prepare_model

def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model
```
Sourced from Ray's V2 `prepare_model` source (the wrap only fires when `world_size > 1`) — **not yet personally reproduced by crashing it** (that's the Debugging Challenge, still open). Consequence: `model.module` only exists once wrapped, so a checkpoint saved without `unwrap()` at 2 workers carries `"module."`-prefixed keys a plain model can't load. `unwrap()` neutralizes this at any world size.

---

**3. Checkpointing — save progress so a crash doesn't cost everything**

`ray_train_stub.py → train_loop_per_worker()`, the checkpoint block:
```python
with tempfile.TemporaryDirectory() as d:
    checkpoint = None
    if rank == 0:
        torch.save({"model": unwrap(model).state_dict(), "opt": opt.state_dict(), "epoch": epoch},
                   os.path.join(d, CKPT_FILE))
        checkpoint = ray.train.Checkpoint.from_directory(d)
    ray.train.report(metrics, checkpoint=checkpoint)
```
Every rank calls `report()` — it's a barrier, skipping it on one rank hangs the others — but only rank 0 attaches a real `Checkpoint`, because under DDP every rank's weights are identical (saving from all of them would just write N redundant copies, confirmed against the checkpoint guide's "filename collisions do not error" note). File-count-before/after verification is the Coding Exercise — **not yet personally run**.

---

**4. Resume — pick back up after a crash or restart**

`ray_train_stub.py → train_loop_per_worker()`, top of the function:
```python
ckpt = ray.train.get_checkpoint()
if ckpt:
    with ckpt.as_directory() as d:
        state = torch.load(os.path.join(d, CKPT_FILE), map_location="cpu")
        model.load_state_dict(state["model"])        # load BEFORE prepare_model
        opt_state, start_epoch = state["opt"], state["epoch"] + 1
```
**Confirmed by a real kill-and-resume cycle** (Lab 3, `world_size=1`): `run_a` killed via `os._exit(1)` right after epoch 2's checkpoint was reported → `trainer.fit()` correctly raised `WorkerGroupError`/`ActorDiedError` (`FailureConfig` defaults to `max_failures=0`, no silent retry — **that large traceback is the correct behavior, not a bug**). Rerunning the identical command printed `RESUMED at epoch 2` and finished through epoch 5. Rerunning it a **third** time — after it had already fully finished — printed `RESUMED at epoch 6`, found no epochs left (`range(6, 6)` is empty), did zero new work, and `result.metrics` just echoed epoch 5's old numbers. Sharper than expected: **reusing a run-name on an already-finished run silently no-ops and still looks like a clean success.**

---

**Remember when revising**
- `use_gpu=True` does nothing useful on Apple Silicon — confirmed: hung 60s with `WorkerGroupStartupTimeoutError`, because Ray never sees an MPS device as a `GPU` resource. Always `use_gpu=False` on a Mac.
- `report()` is a collective — skipping it on even one rank hangs the whole run instead of erroring.
- A **reused `--run-name`** silently resumes old state instead of starting fresh — confirmed sharper than expected: reusing an *already-finished* run's name does zero new work and still exits looking successful.
- `model.module` only exists once `world_size > 1` — unwrap conditionally before every save (mechanism sourced from Ray's V2 code; the actual failure hasn't been personally reproduced yet — Debugging Challenge still open).
- Build the optimizer **after** `prepare_model()` — harmless on CPU, required on GPU (`.to(device)` creates new parameter tensors first).
- `equal=True` sharding drops leftover rows — confirmed on 21 rows / 2 workers (drops 1). Never rely on *which* rows land where, only that the split is disjoint and balanced.
- Array-valued columns (`observation.state`, `action`) come back as 1-D object arrays only when something forces `batch_format="numpy"` — Week 3 Day 1's `map_batches(batch_format="numpy")` bug, and (confirmed this session) any `iter_torch_batches` call where **you** pass a `collate_fn`. With **no** `collate_fn`, `iter_torch_batches()` reads Arrow tensor-extension columns directly via an internal `DefaultCollateFn`/`batch_format="pyarrow"` path and never hits the object-dtype array at all — confirmed by a real run returning correct `(32, 14)` tensors with zero conversion code.
- A plain function passed as `collate_fn` is deprecated as of Ray 2.47 — confirmed live (`RayDeprecationWarning` on this exact run). Subclass `NumpyBatchCollateFn` (or `ArrowBatchCollateFn` / `PandasBatchCollateFn` for those batch formats) instead, and pass an instance, not the bare class.
- `mean_loss = loss_sum / n_batches` (as computed in `train_loop_per_worker`) is an average of per-batch averages, not a true per-row average — biased whenever the last batch is a different size than the rest, which it always is unless the row count divides evenly by `batch_size`.

**When to actually reach for this:** dataset too big for one machine's disk, need CPU-decode and GPU-train to scale independently, or need worker-failure retries. Otherwise `accelerate launch lerobot-train` (2–8 GPUs, one box) or plain `lerobot-train` (fits on one GPU) is simpler and is what you should default to.

**Interview framing:** "I isolated each of Ray Train's four mechanisms on a toy before trusting it in a real loop, and two of my four real runs surfaced something the plan didn't anticipate — a live deprecation warning traced to a genuine internal `batch_format` branch, and a 'reused run-name' failure mode subtler than the one I'd planned to demonstrate. I still have the DDP-wrap checkpoint bug and the rank-0 file-count check to personally reproduce before I'd call this fully proven, not just planned."

---

## Recurring Threads

Patterns that showed up more than once, worth having ready as a single answer rather than three separate ones:

- **Global vs. local indexing after a filter.** Week 1 Day 1 → named explicitly Day 4 → the same shape reappears as the contiguity assertion in Week 3 Day 2's partitioning code. Filtering changes what a positional index *means*.
- **Never trust exact equality on timestamps.** Week 1 Day 3's sync detectors → Week 2 Day 1's float32-precision gap numbers. Always a tolerance, never `==`.
- **Chunking/storage parameters are never semantic.** MCAP's `chunk_size` (Week 2 Day 1) and HDF5 chunk shape (Week 2 Day 5) both only affect seek performance and file layout, never message content.
- **Verify library behavior directly — via a debug print or the real source — rather than trusting documentation.** Week 3 Day 1's tensor-shape bug and Week 3 Day 2's decord/PyAV pivot both hinged on this. Week 3 Day 3 adds a third confirmation: `iter_torch_batches()`'s behavior branches on whether you pass a `collate_fn` at all (`batch_format="pyarrow"` internally with none, `"numpy"` the moment you supply your own) — and a bare-function `collate_fn` works today while printing a real deprecation warning for tomorrow.
### Day 4 — Spark for the Tabular Slice, and the Seam Between Two Engines

**The core idea.** Not every table in a robotics data platform needs a
GPU-adjacent engine. Fleet facts at episode grain -- counts, duration,
sync errors, firmware -- are plain aggregates over Parquet: Spark's job,
the same warehouse pattern with episode_index as the grain. The real
skill is the SEAM: Ray Data produces per-FRAME signals that need pixels;
Spark rolls them up to episode grain, joins on a key both engines emit
identically, casts types in exactly one place (normalize_keys).

**Four mundane bugs, all hit by running:** Spark won't expand `~`
(confirmed FileNotFoundException); LeRobot's dotted keys (observation.
state, next.done) need backticks or Spark parses a struct path (confirmed
UNRESOLVED_COLUMN.WITH_SUGGESTION, whose "did you mean" literally echoes
the typed name back); max(ts)-min(ts) is one frame period short (29.98
vs 30.0) -- demonstrated intentionally in lab1_episodes.py, then
corrected via num_frames/FPS once fps is loaded from meta/info.json; the
partition trap (partitionBy strips the column from every file, confirmed
via pq.read_schema).

**Real numbers.** 85 episodes, reconcile mismatches: 0 (both runs). sync
flagged: 0/85 (synthetic timestamps prove the machinery, not the
phenomenon); max_gap_s=0.020000458, identical across every episode.
Ray outlier flag dataset-wide: mean=2.1567 std=0.7756, 0/127500 flagged
at 3-sigma -- sanity-checked against per-episode max_action_magnitude
(3.5-4.0, safely below mean+3std=4.483): a genuine well-behaved-dataset
finding. Lab 0 JVM startup: 3.69s.

**A real bug found from running the pipeline twice, not from a script:**
ray_bronze_ingest_simplified.py's write never clears its output dir --
ray_output_day1/ silently accumulated 4 duplicate copies. Harmless this
time only because SUM-of-zero and MAX are duplication-invariant.

**A real, deeper bug found by reading Ray's actual source, after
disproving two of my own theories with real runs:** FailureConfig's
auto-retry works exactly as documented, but a hard os._exit() right
after report() can deterministically lose that exact report --
reproduced across 5 runs. Root cause, confirmed in controller.py: the
controller retrieves reports via its OWN periodic poll, not
synchronously when report() returns. A worker-side sleep had zero
effect -- the gap is cross-process, no training-loop fix exists.

**Interview framing:** "I don't pick an engine by familiarity -- I ask
whether one row of output needs a decoded frame or a model pass. I also
don't stop at 'it works': when a checkpoint-resume test kept silently
redoing the same epoch, I disproved my own first fix empirically before
reading Ray Train's installed source to find the real mechanism -- the
controller polls for reports on its own schedule, so a hard kill can
lose the most recently reported checkpoint before the controller ever
sees it. That's the kind of fault-tolerance edge case that matters once
training moves from a laptop to a real GPU cluster."

### Day 4 — Spark for the Tabular Slice, and the Seam Between Two Engines

**The core idea.** Not every table in a robotics data platform needs a
GPU-adjacent engine. Fleet facts at episode grain -- counts, duration,
sync errors, firmware -- are plain aggregates over Parquet: Spark's job,
the same warehouse pattern with episode_index as the grain. The real
skill is the SEAM: Ray Data produces per-FRAME signals that need pixels;
Spark rolls them up to episode grain, joins on a key both engines emit
identically, casts types in exactly one place (normalize_keys).

**Four mundane bugs, all hit by running:** Spark won't expand `~`
(confirmed FileNotFoundException); LeRobot's dotted keys (observation.
state, next.done) need backticks or Spark parses a struct path (confirmed
UNRESOLVED_COLUMN.WITH_SUGGESTION, whose "did you mean" literally echoes
the typed name back); max(ts)-min(ts) is one frame period short (29.98
vs 30.0) -- demonstrated intentionally in lab1_episodes.py, then
corrected via num_frames/FPS once fps is loaded from meta/info.json; the
partition trap (partitionBy strips the column from every file, confirmed
via pq.read_schema).

**Real numbers.** 85 episodes, reconcile mismatches: 0 (both runs). sync
flagged: 0/85 (synthetic timestamps prove the machinery, not the
phenomenon); max_gap_s=0.020000458, identical across every episode.
Ray outlier flag dataset-wide: mean=2.1567 std=0.7756, 0/127500 flagged
at 3-sigma -- sanity-checked against per-episode max_action_magnitude
(3.5-4.0, safely below mean+3std=4.483): a genuine well-behaved-dataset
finding. Lab 0 JVM startup: 3.69s.

**A real bug found from running the pipeline twice, not from a script:**
ray_bronze_ingest_simplified.py's write never clears its output dir --
ray_output_day1/ silently accumulated 4 duplicate copies. Harmless this
time only because SUM-of-zero and MAX are duplication-invariant.

**A real, deeper bug found by reading Ray's actual source, after
disproving two of my own theories with real runs:** FailureConfig's
auto-retry works exactly as documented, but a hard os._exit() right
after report() can deterministically lose that exact report --
reproduced across 5 runs. Root cause, confirmed in controller.py: the
controller retrieves reports via its OWN periodic poll, not
synchronously when report() returns. A worker-side sleep had zero
effect -- the gap is cross-process, no training-loop fix exists.

**Interview framing:** "I don't pick an engine by familiarity -- I ask
whether one row of output needs a decoded frame or a model pass. I also
don't stop at 'it works': when a checkpoint-resume test kept silently
redoing the same epoch, I disproved my own first fix empirically before
reading Ray Train's installed source to find the real mechanism -- the
controller polls for reports on its own schedule, so a hard kill can
lose the most recently reported checkpoint before the controller ever
sees it. That's the kind of fault-tolerance edge case that matters once
training moves from a laptop to a real GPU cluster."

### Day 4 — Spark for the Tabular Slice, and the Seam Between Two Engines

**The core idea.** Not every table in a robotics data platform needs a
GPU-adjacent engine. Fleet facts at episode grain -- counts, duration,
sync errors, firmware -- are plain aggregates over Parquet: Spark's job,
the same warehouse pattern with episode_index as the grain. The real
skill is the SEAM: Ray Data produces per-FRAME signals that need pixels;
Spark rolls them up to episode grain, joins on a key both engines emit
identically, casts types in exactly one place (normalize_keys).

**Four mundane bugs, all hit by running:** Spark won't expand `~`
(confirmed FileNotFoundException); LeRobot's dotted keys (observation.
state, next.done) need backticks or Spark parses a struct path (confirmed
UNRESOLVED_COLUMN.WITH_SUGGESTION, whose "did you mean" literally echoes
the typed name back); max(ts)-min(ts) is one frame period short (29.98
vs 30.0) -- demonstrated intentionally in lab1_episodes.py, then
corrected via num_frames/FPS once fps is loaded from meta/info.json; the
partition trap (partitionBy strips the column from every file, confirmed
via pq.read_schema).

**Real numbers.** 85 episodes, reconcile mismatches: 0 (both runs). sync
flagged: 0/85 (synthetic timestamps prove the machinery, not the
phenomenon); max_gap_s=0.020000458, identical across every episode.
Ray outlier flag dataset-wide: mean=2.1567 std=0.7756, 0/127500 flagged
at 3-sigma -- sanity-checked against per-episode max_action_magnitude
(3.5-4.0, safely below mean+3std=4.483): a genuine well-behaved-dataset
finding. Lab 0 JVM startup: 3.69s.

**A real bug found from running the pipeline twice, not from a script:**
ray_bronze_ingest_simplified.py's write never clears its output dir --
ray_output_day1/ silently accumulated 4 duplicate copies. Harmless this
time only because SUM-of-zero and MAX are duplication-invariant.

**A real, deeper bug found by reading Ray's actual source, after
disproving two of my own theories with real runs:** FailureConfig's
auto-retry works exactly as documented, but a hard os._exit() right
after report() can deterministically lose that exact report --
reproduced across 5 runs. Root cause, confirmed in controller.py: the
controller retrieves reports via its OWN periodic poll, not
synchronously when report() returns. A worker-side sleep had zero
effect -- the gap is cross-process, no training-loop fix exists.

**Interview framing:** "I don't pick an engine by familiarity -- I ask
whether one row of output needs a decoded frame or a model pass. I also
don't stop at 'it works': when a checkpoint-resume test kept silently
redoing the same epoch, I disproved my own first fix empirically before
reading Ray Train's installed source to find the real mechanism -- the
controller polls for reports on its own schedule, so a hard kill can
lose the most recently reported checkpoint before the controller ever
sees it. That's the kind of fault-tolerance edge case that matters once
training moves from a laptop to a real GPU cluster."
