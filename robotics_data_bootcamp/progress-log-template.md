# Progress Log — Robotics Data Engineering

Update this after every session, then re-upload it to the project's
Knowledge Base (replacing the old version). This is how a brand-new chat
knows what you've already done.

## Environment
- Machine: MacBook M4 Max
- LeRobot version: (run `python -c "import lerobot; print(lerobot.__version__)"` and fill in)
- MPS available: (fill in — output of the torch.backends.mps check)
- Primary dataset in use: lerobot/aloha_mobile_cabinet (fallback: lerobot/pusht)
- Repo: robot-data-forge

## Week 1

### Day 1 — status: complete
- Built: episode_card.py (CLI only — no describe_episode(ds, idx) function was ever
  factored out of main(); everything's inline. Matters for Day 5, see below.)
- Key gotcha hit: ds[args.episode] used to fetch action/state dims after
  constructing ds = LeRobotDataset(repo_id, episodes=[args.episode]) — that filter
  re-indexes ds locally to 0..length-1, so ds[args.episode] grabs frame number
  args.episode WITHIN that one episode, not "episode args.episode." Silent for
  action/state dims (shape is always 14 regardless of frame); would IndexError on
  a short, late episode. Same local/global confusion Day 4 later named explicitly —
  it was already latent here, just not caught yet.
- Commit: [fill in]
- Open questions carried forward:
  - Needs a describe_episode(ds, idx) -> dict refactor before Day 5 can import it.
  - num_frames currently reads meta.total_frames (whole dataset) instead of this
    episode's length; cameras only shows camera_keys[0] instead of all cameras.


### Day 2 — status: complete
- Built: sensor_inventory.py (coverage matrix across real episodes, v3.0-correct indexing) + fps verification helper
- Key gotcha hit: [fill in — expected candidate: ds[i] != episode i under v3.0's shard-based storage]
- Commit: day2: sensor_inventory.py — v3.0-correct episode indexing, coverage + degenerate checks
- Open questions carried forward: 

### Day 3 — status: complete
- Built: sync_checker.py (tiered: hf_dataset-only tabular checks on all episodes,
  streaming visual checks on a sample; dropped/duplicate/held-frame detectors,
  synthetic-fault tested)
- Key gotcha hit: naive per-frame extraction ([ds[i] for i in range(...)]) decodes
  every camera for every frame just to read a timestamp scalar — OOM'd Colab.
  Fixed via ds.hf_dataset (Arrow/Parquet-backed, tabular-only, no video decode).
  Second finding: aloha_mobile_cabinet exposes one shared `timestamp` per row,
  not separate per-camera clocks — cross-sensor drift only observable indirectly
  via held-frame runs at this layer.
- Commit: day3: sync_checker.py — tiered tabular/visual checks, RAM-safe
  hf_dataset extraction, synthetic-fault tests
- Open questions carried forward:

### Day 4 — status: complete
- Built: trajectory_checker.py (RAM-safe joint-space trajectory extraction via
  ds.hf_dataset; action-following anomaly detector, absolute-target formula, not
  delta; arm-flip detector; --plot mode using range(len(motor_names)) rather than
  a hardcoded 14 — good habit, keep it)
- Key gotcha hit: dataset_from_index/dataset_to_index are GLOBAL, but
  LeRobotDataset(repo_id, episodes=[...]) re-indexes hf_dataset LOCALLY —
  positional slicing with global indices against a filtered dataset silently
  returns an empty result (confirmed against real, currently-open LeRobot GitHub
  issues #816/#1783/#1895/#2678). Fixed by filtering hf_dataset on the
  episode_index column value instead of by position.
- Commit: [fill in]
- Open questions carried forward:

### Day 5 (Capstone) — status: complete
- Built: week1_report.py (full pipeline: describe_episode + sensor_coverage_matrix
  + sync detectors + action_following_error merged by episode_index; gated health
  scorer with hard-exclude on schema failure and separate insufficient_data
  decision for episodes trajectory_checker couldn't score; markdown report +
  CI exit code)
- Key gotcha hit: episode_card.py never had describe_episode() factored out on
  Day 1 -- had to refactor before the capstone's "import and call" pattern
  worked. Also: naive equal-weighted health score let a schema-broken episode
  score 0.69 (looks fine-ish) -- fixed with a hard gate that forces schema
  failures to 0.0 and marks them "excluded" rather than folding them into the
  weighted average with the other three checks.
- Commit: day5: week1_report.py -- full pipeline + capstone health report
- Open questions carried forward:
  - sensor_inventory.py / sync_checker.py / trajectory_checker.py function
    signatures assumed in week1_report.py's imports haven't been confirmed
    against my actual files yet -- verify before first real run.
  - action_error_ceiling (0.15 rad) is a placeholder -- tune against this
    dataset's real action_following_error distribution, not assumed.

## Week 1 — status: COMPLETE (Days 1-5)

## Week 2

### Day 1 — status: complete
- Built: mcap_from_lerobot.py (LeRobot v3.0 episode -> MCAP; episode range
  resolved by episode_index VALUE, single batched ds.hf_dataset slice, JSON
  schema + provenance metadata record, --split-topics, --time-base
  episode|global, --compression, --chunk-size) + mcap_inspect.py (two-tier:
  cheap get_summary() pass and full iter_messages() pass; per-topic count /
  duration / implied Hz / median-relative gap flags / non-monotonic count;
  non-zero exit for CI)
- Key gotcha hit: LeRobot `timestamp` is float32 seconds FROM EPISODE START and
  resets to 0.0 every episode; MCAP `log_time` is int64 ns on a file-wide
  monotonic clock. Writing `int(timestamp * 1e9)` straight through for a
  multi-episode file is structurally valid, opens without error, and is
  silently wrong: episodes interleave, message_start/end_time report the
  longest episode instead of the sum, and the implied rate comes out as
  fps x n_episodes. Fixed with a cumulative per-episode offset
  (episode_time_offset_ns from declared length/fps).
- Second finding: float32 timestamps mean the derived ns gaps are never
  identical -- on a clean 1500-frame 50Hz episode only 2/1499 gaps land
  exactly on 20,000,000 ns (spread 19,998,550-20,000,458). Confirms Week 1's
  "never == on timestamps" rule with a number. gap_report thresholds relative
  to the median, and treats non-monotonic transitions as a separate fault
  from large gaps.
- Third finding: chunk_size changes chunk_count and seek performance only,
  never message content -- storage decision, not a semantic one. Same lesson
  as HDF5 chunking, one layer earlier in the pipeline.
- Commit: [fill in] week2-day1: mcap_from_lerobot.py + mcap_inspect.py
- Open questions carried forward:
  - --time-base global writes fictional log_times. If aloha_mobile_cabinet
    carries a real recording date anywhere in meta/, prefer a real epoch so
    the file is joinable against external logs. Not checked yet.
  - Protobuf encoding comparison (stretch goal) not run -- size/write/read
    numbers still TODO.
  - mcap_inspect.py does not yet cross-check st.message_count (a writer claim)
    against the count from iteration. Add before it's used as a real gate.

### Day 2 — status: complete (rebuilt after 4-round debugging log)
- Built: rosbag2_inspect.py (AnyReader-based, custom-typestore-aware;
  open_and_inspect() reports an unreadable bag as a clean non-zero exit
  instead of crashing; --convert-to-mcap-report flag)
- Approach change, mid-day: original plan (wrap Day 1's custom-JSON-schema
  episode0.mcap in a hand-rolled metadata.yaml, read with AnyReader) is
  fundamentally not viable -- confirmed across FOUR independent validation
  errors, not four independent bugs:
    1. metadata.yaml serialization_format must be "cdr"
       (ReaderError: Serialization format {'json'} is not supported.)
    2. [unrelated bug, not architectural: .rename() consumed its own input
       on a rerun -- ReaderError: Some database files are missing]
    3. MCAP Header.profile must be "ros2"
       (ReaderError: Profile is not ros2.)
    4. every schema's encoding must be ros2msg/ros2idl, checked while
       cataloging ALL schemas during open(), before .connections exists
       (KeyError: 'jsonschema', from rosbags/rosbag2/storage_mcap.py's
       get_msgdef())
  (1) and (3) were patchable with deliberate lies; (4) is not, because it's
  reading a real field from the real file, not trusting a manifest claim.
  rosbag2_storage_mcap's own docs confirm the scope: "self-contained MCAP
  files that include all the definitions needed to decode custom ROS 2
  messages" -- custom ROS types, never arbitrary schemas.
- Rebuilt Lab 1/Lab 2 around the legitimate path: defined a real ROS 2
  message type (lerobot_msgs/msg/JointState: frame_index int32,
  state/action float32[14]) via rosbags.typesys.get_types_from_msg,
  registered it in a typestore, wrote a genuine rosbag2 bag with
  rosbags.rosbag2.Writer(storage_plugin=StoragePlugin.MCAP) -- correct
  metadata.yaml, profile, serialization_format, and schema encoding, all
  produced by the library instead of guessed by hand.
- Key gotcha (NEW, found while rebuilding, not from the 4-stage saga):
  rosbags-generated message classes need numpy arrays for array fields --
  plain Python lists fail inside serialize_cdr with AttributeError: 'list'
  object has no attribute 'view' (confirmed against a real reported issue
  against the library). Fixed with np.array(..., dtype=np.float32).
- Second gotcha: a custom type only exists in the Python process that
  registered it -- AnyReader does not auto-discover a bag's embedded custom
  message definitions. Pass default_typestore= with the type pre-registered
  (what Lab 2 does), or pull connection.msgdef and register it manually.
- [PENDING YOUR RUN -- none of this has been executed against the real
  rosbags/mcap packages, only syntax-checked and logic-tested against
  mocks. Fill in: did Lab 1 write successfully? Did Lab 2's connections
  print the real msgtype? Did deserialize() actually return a usable
  message? If anything diverges, that's the next real finding.]
- Commit: week2-day2: rosbag2_inspect.py + a real custom-typed rosbag2 bag
- Open questions carried forward:
  - Confirm the full Lab 1 -> Lab 2 round trip actually runs clean.
  - rosbags-dataframe's get_dataframe() should now work against a real
    registered type -- reasoned, not run.
  - If the custom-type approach hits yet another snag: fall back to a real
    downloaded public rosbag2 sample bag to isolate whether the problem is
    AnyReader's behavior in general or something specific to this build.

### Day 3 — status: complete
- Built: rlds_schema_diff.py (compares observation/action schemas between two OXE
  sub-datasets -- key set diffs, shared-key shape/dtype mismatch report, leaf-summed
  action dimensionality, free per-side episode totals) + rlds_inspect.py (single-
  dataset profiler -- free corpus-wide total from builder.info, sampled N-episode
  step-count min/median/max, observation/action key report)
- Approach change, mid-session: originally planned to load OXE sub-datasets via the
  jxu124/OpenX-Embodiment HF mirror (per the course's own Week 2 Day 3 draft).
  Confirmed dead on any current `datasets` install -- `datasets` 4.0.0 (9 Jul 2025)
  permanently removed script-based dataset loading, and jxu124's repo (script-based,
  never converted to Parquet) fails with `RuntimeError: Dataset scripts are no
  longer supported`. Rebuilt every load path around `tfds.builder_from_directory`
  reading directly from the public gs://gresearch/robotics bucket instead --
  confirmed against the official open_x_embodiment repo and DROID's own docs.
- Key gotcha hit: hardcoded "0.1.0 default version, two named exceptions" rule
  (sourced from a 2023-era blog post covering the original OXE release) doesn't
  hold for aloha_mobile (a 2024 addition) -- real version is 0.0.1. Confirmed via
  gsutil ls / tf.io.gfile.listdir directly against the bucket, not assumed.
- Second gotcha: naive version auto-discovery (sorted(listdir(...))[-1]) picked a
  GCS placeholder marker object (0.0.1_$folder$) over the real 0.0.1/ directory,
  because the placeholder string sorts after the real one alphabetically. Fixed
  by filtering on the trailing "/" that tf.io.gfile.listdir puts on real
  subdirectories only.
- Third gotcha: reading even one aloha_mobile episode crashed the Colab session's
  RAM. Root cause: TFDS decodes every Image feature by default regardless of use;
  aloha_mobile's whole-body-manipulation episodes are long, so one episode meant
  decoding 3 cameras x every step. Fixed with tfds.decode.PartialDecoding to skip
  the camera fields entirely, plus tfds.ReadConfig(override_buffer_size=1024) and
  tf.data.Options() autotune/prefetch-disabling from TF's own "tf.data uses all
  my RAM!" doc.
- Real finding (ties back to course.md): aloha_mobile's RLDS action is
  float32[16] vs lerobot/aloha_mobile_cabinet's float32[14] -- first concrete
  evidence for course.md's claim that a base_action field was dropped in the
  LeRobot conversion. observation.state matches at float32[14] and camera names
  match exactly (cam_high/cam_left_wrist/cam_right_wrist), supporting shared
  lineage. Episode counts don't match (276 vs 85) -- aloha_mobile aggregates
  multiple Mobile ALOHA tasks, not just cabinet -- compare by schema/content,
  never by episode index.
- Commit: [fill in] week2-day3: rlds_schema_diff.py + rlds_inspect.py -- native
  GCS loading, verified schemas, RAM-safe decoding for long-episode datasets
- Open questions carried forward:
  - Confirm the RAM-safe Debugging Challenge fix actually completes end-to-end --
    not yet verified in a real run.
  - Stretch goal not done: pull a real aloha_mobile cabinet-task episode (match
    by language_instruction, not index) and diff actual action/state VALUES
    against a lerobot/aloha_mobile_cabinet episode -- confirms the base_action
    split at the value level, not just shape.
  - week2-robotics-data-bootcamp.md (project knowledge) still has the ORIGINAL
    jxu124-mirror-based Day 3 draft -- replace before a future session reads it
    as source of truth (see item 5 below).
  - oxe_builder / rlds_inspect.py / rlds_schema_diff.py don't yet default to the
    RAM-safe decoders -- currently manual opt-in per dataset. Worth a
    --skip-images flag if this bites again on another long-episode dataset.

### Day 4 — status: complete
- Built: verify_conversion.py (episode_index-VALUE-matched, per-field element-wise
  diff between two revisions/roots of a LeRobot dataset, tolerance-gated, non-zero
  exit on drift) + format_migration_gate.py (generalized version -- Hub revisions
  OR local DVC-style root paths, reusable beyond this week's v2.1/v3.0 check)
- Key gotcha hit: [fill in -- expected candidates: (1) which of the two conversion
  script module paths (lerobot.scripts.* vs lerobot.datasets.v30.*) your installed
  version actually resolves; (2) whether lerobot/aloha_mobile_cabinet itself
  carries a v2.1 tag or the svla_so101_pickplace fallback was needed; (3) whether
  LeRobotDataset(repo_id, root=...) works as written for the local-path branch of
  format_migration_gate.py -- untested this session, reasoned from
  LeRobotDatasetMetadata's documented root= param only]
- Verified against real sources this session (not just docs): huggingface/lerobot#2689
  confirmed real and open (aloha_sim_insertion_scripted + pi0.5, erratic joint
  actions post-conversion); real meta/info.json pulled for aloha_mobile_cabinet
  itself (codebase_version v3.0, 85 episodes, 127500 frames, fps 50, av1 video
  codec, chunks_size 1000, data_path/video_path template strings); revision="v2.1"
  Hub pattern confirmed via huggingface/lerobot#1998; converter write-access
  requirement confirmed via huggingface/lerobot#2446.
- Commit: [fill in] week2-day4: verify_conversion.py + format_migration_gate.py
- Open questions carried forward:
  - Neither script has been run against a real install yet -- syntax-checked via
    py_compile only, zero network access this session. First real run is the
    actual test.
  - chunks_size's exact unit (episodes vs. files vs. bytes) not confirmed --
    Lab 1 asks you to check meta.episodes for a chunk/file-index column and
    report back what you find.
  - meta/tasks.parquet vs meta/tasks.jsonl -- current HF docs prose says one,
    a real recent Hub commit shows the other. Note which one your actual
    dataset has.

### Day 5 — status: complete [reconstructed -- this day's own session log was
thin; rebuilt from what survived plus revision-notes.md's own reconstructed
Day 5 section. Flag anything below that doesn't match what you remember.]
- Built: inspect_hdf5.py (walks an ALOHA-native HDF5 episode file; reports
  shape/dtype/chunk-shape/compression codec per dataset; flags whether
  observations/effort and base_action are present)
- Key finding: raw HDF5 capture includes qvel and base_action, but the
  LeRobot conversion only carries action, state (<-qpos), and effort
  forward -- qvel/base_action never reach the Parquet layer. Practical
  consequence: reconstructing an HDF5 file FROM a LeRobot dataset cannot
  repopulate them; that data is already gone by the time it reaches you.
  Independently corroborates Day 3's aloha_mobile RLDS action=float32[16]
  vs. LeRobot's float32[14] finding -- two unrelated checks pointing at
  the same gap.
- Second finding: HDF5 per-dataset chunking/compression is the same
  "storage decision, not a semantic one" lesson as MCAP's chunk_size
  (Day 1) -- one layer earlier in the pipeline.
- Commit: [fill in]
- Open questions carried forward:
  - This reconstruction hasn't been re-verified against a fresh run --
    confirm inspect_hdf5.py's actual output next time it's touched.

## Week 3

### Day 1 — status: complete
- Built: tensor_utils.py (shared `to_2d_float_array()` helper, imported by every
  script that touches `action`) + lab1_ray_data_intro.py (count()'s cost model:
  footer-metadata fast path on a bare read vs. full execution once map_batches
  is chained) + lab2_action_magnitude_flag.py (action magnitude + deliberately
  naive per-block outlier flag, kept as a teaching artifact) +
  debug_count_behavior.py (two open empirical questions Ray's docs don't
  resolve) + ray_bronze_ingest.py (dataset-wide mean/std via one
  ds.aggregate() call — the real fix for Lab 2's per-block flaw) +
  mini_project_ray_vs_spark_timing.py (same transform, both engines, timed) +
  three practice docs (toy-data Ray Data walkthrough, real-dataset query
  practice with select_columns/filter/sort/groupby, file-group partitioning
  deep dive) + a cheat sheet (Google Doc, robotics/ folder in Drive).

- Key gotcha, REAL root cause (found via a debug print, not guessed): Ray Data
  represented aloha_mobile_cabinet's `action` column as a 1-D dtype=object
  array of per-row arrays, NOT a (batch_size, 14) 2D tensor — despite Ray's
  own docs describing fixed-shape tensor columns as always coming back as
  regular ndarrays. `np.linalg.norm(action, axis=-1)` on that shape silently
  reduced the wrong axis (across rows, not across the 14 joints) instead of
  erroring — produced a wrong-length output ("expected 256 got 14") that
  looked like an off-by-something bug. Reproduced standalone with plain numpy
  to confirm the exact mechanism.

- Two wrong hypotheses tried first, both disproven by the SAME error
  surviving the "fix" — worth remembering as a debugging pattern, not just a
  footnote: (1) a LimitOperator/take_batch() interaction theory, disproven
  when materialize() before take_batch() changed nothing; (2) an in-place
  batch-dict-mutation theory, disproven when switching to `{**batch, ...}`
  changed nothing either. Identical error surviving a fix is the signal the
  theory is wrong, not the fix. Real fix: to_2d_float_array() in
  tensor_utils.py, np.stack()s the object array into a real 2D array before
  any math touches it.

- Course-material correction, verified against live Ray docs (docs.ray.io,
  Ray 2.56.0): ds.count() on a BARE read_parquet() is NOT a full scan — Ray
  reads Parquet footer metadata, same trick Spark/DuckDB use. Still costs
  more than LeRobotDatasetMetadata.total_frames (scales with file count vs.
  zero file opens), but the original course draft's "count() will be slow"
  assumption was wrong. Fast path breaks the instant ANY map_batches is
  chained before count().

- Real column names for meta.episodes confirmed via LeRobot's own GitHub
  source (lerobot_dataset.py, get_video_file_path):
  videos/{video_key}/chunk_index, videos/{video_key}/file_index per camera,
  dataset_from_index/dataset_to_index for row ranges. Also already sitting in
  Week 1 Day 2's own notebook output — this was re-deriving something already
  known, not new information. Same for "action is an absolute joint target,
  not a delta" (Week 1 Day 4, leader-follower teleoperation).

- Real file-group partitioning result on aloha_mobile_cabinet (technique
  sourced from Anyscale, "Optimizing VLA Fine-Tuning Performance with
  LeRobot Datasets and Ray," Feb 10 2026): 85 episodes -> 4 partitions, not
  1. The 3 cameras roll over to their 2nd video file at three DIFFERENT
  episodes (cam_high at 66, cam_right_wrist at 79, cam_left_wrist at 82) —
  confirms combining ALL cameras into one partition key is necessary; a
  single-camera key gives a wrong, too-optimistic grouping. ~21x fewer video
  opens than naive per-episode parallelization (255 -> 12) — smaller than
  DROID's published 135x, honestly explained by dataset size, not a failed
  replication.

- Real Ray Data API confirmed and applied (Loading Data / Performance Tips /
  read_parquet docs): columns=[...] on read_parquet() pushes column
  selection to the file scan (projection pushdown); ds.filter(expr=col(...)
  == lit(...)) from ray.data.expressions is pushdown-capable, a plain lambda
  filter is not. Applied to ray_bronze_ingest.py, then intentionally
  reverted — see next item.

- **STATE NOTE FOR NEXT SESSION — read this before assuming anything about
  ray_bronze_ingest.py:** the file currently in the repo is the SIMPLIFIED
  core version — no argparse, no columns=[...] pruning, no expression-based
  filter, no select_columns() output cleanup. All four were added once,
  confirmed correct and properly sourced, then deliberately stripped back
  out mid-session because the accumulated complexity became more than a
  first pass at this material should reasonably ask of anyone. The
  production-polish version exists only in this session's transcript, not
  in the working file. Re-introduce incrementally — one piece at a time,
  the same way Lab 2's fixes were folded in — only once the simplified
  version feels fully solid. Don't restore it all at once.

- Pedagogical finding worth carrying into Day 2 planning: Ray Data's lazy
  execution model, the batch-as-dict-of-arrays mental model, and the tensor
  shape gotcha needed substantially more ground-up scaffolding (toy data
  before real data, real traced output at every single step, one new
  concept introduced at a time) than Weeks 1-2's format-reader work did.
  Day 2 should default to this same slower, concrete-first pace rather than
  the original course draft's denser style — introduce new files/concepts
  one at a time, not stacked together.

- Commit: week3-day1: tensor_utils.py + lab1/lab2 + debug_count_behavior.py
  + ray_bronze_ingest.py (simplified) + mini_project_ray_vs_spark_timing.py
  + 3 practice docs + Drive cheat sheet

- Open questions carried forward:
  - Debugging Challenge's two open empirical questions never confirmed
    against a real run: does count() get cheaper on a repeated call? does
    udf_modifying_row_count=False measurably change count()'s timing?
  - mini_project_ray_vs_spark_timing.py never run for real — Ray vs. Spark
    numbers on this dataset are still unknown.
  - The production-polish version of ray_bronze_ingest.py (column pruning,
    expression filter, clean output) is correct and sourced, but not yet
    re-merged into the working file (see STATE NOTE above).
  - Optional PyAV video-opening stretch goal from the partitioning exercise
    (actually opening one video per partition) not attempted — new
    dependency, deliberately deferred.
  - Real dataset-wide mean/std and flag rate from ray_bronze_ingest.py (3.0
    std threshold) not yet reported back from an actual run on the full
    127,500 frames.

### Day 2 — status: complete
- Built: partition_utils.py (build_file_group_partitions(); get_episode_frame_counts()
  -- the Ray Data version, kept for reference but blocked, see gotcha;
  get_episode_frame_counts_torchcodec() -- the working replacement, CONFIRMED) +
  day2_lab_exploration.py (Labs 1a/1b/2 + Debugging Challenge, run top-to-bottom,
  all confirmed) + partition_report.py --repo-id [--verify-read] [--camera]
  (naive-vs-file-group open counts + real video verification, CONFIRMED)
- Key gotcha hit: ray.data.read_videos() -- the API Day 2's plan was built
  around -- requires `decord` internally (confirmed via traceback:
  ray/data/_internal/datasource/video_datasource.py calls
  _check_import(module="decord")). decord imported fine as a Python package,
  but its compiled binary (libdecord.dylib) failed to dlopen because it's
  linked against FFmpeg's libavformat.61, which a Homebrew ffmpeg upgrade
  (now at 8.1) no longer ships. decord also confirmed effectively
  unmaintained (last real commit years old; own docs still reference
  ffmpeg 4). Fixed by dropping read_videos()/decord and using torchcodec instead --
  already installed as LeRobot's own decode backend on this machine.
  get_frames_played_in_range(start_seconds, stop_seconds) does the [lo, hi)
  timestamp-window slicing natively. CONFIRMED working end to end, then
  superseded before the day closed: checked what real practitioners
  actually do for this exact job -- Ray's own Data maintainer recommends a
  custom Datasource over read_videos() for open-source Ray, and Anyscale's
  own published reference pipeline for LeRobot-on-Ray uses PyAV, not
  decord or torchcodec. Rewrote get_episode_frame_counts_pyav() to match:
  open the file once, seek once to the first episode's from_timestamp,
  decode forward continuously across the whole group -- re-confirmed
  expected==found on all three episodes after the rewrite. Bonus: my own
  build_file_group_partitions() turned out structurally near-identical to
  Anyscale's own published partitioning function, derived independently
  before reading theirs.
- CONFIRMED by a real run: all three episodes in the smallest partition
  (79, 80, 81) came back expected==found (1500==1500) exactly -- no
  off-by-one at the [lo, hi) timestamp boundary. The Debugging Challenge's
  toy contiguity break also fired correctly.
- Real numbers confirmed end to end: 85 episodes -> 4 partitions, 255 -> 12
  video opens, 21.2x reduction -- matches Day 1's practice-doc numbers, now
  from tested code instead of a notebook exploration.
- New finding, not anticipated in planning: --verify-read printed "file
  reports 28500 frames total (partition metadata says 4500)" -- looks like a
  mismatch, isn't one. cam_high's file-001.mp4 physically contains ALL of
  episodes 66-84 (19 episodes x 1500 = 28500, matching cam_high's own
  rollover point from Day 1) because cam_high only changes files once more
  after episode 66. But cam_right_wrist and cam_left_wrist roll over at 79
  and 82 respectively -- INSIDE that same range -- so the combined 3-camera
  partition key splits episodes 66-84 into three separate partitions
  (66-78, 79-81, 82-84) that all happen to share the same physical cam_high
  file. verify_read() only ever asks for one partition's timestamp slice
  out of that larger file and gets exactly that -- correct behavior, just a
  confusingly-worded print statement. Direct confirmation of Day 1's
  "single-camera key gives a wrong, too-optimistic grouping" finding.
- Also observed, not yet root-caused: two `objc[...]: Class AVFFrameReceiver
  is implemented in both ... av/.dylibs/libavdevice.61... and
  .../ffmpeg/8.1/.../libavdevice.62...` warnings on every run. Likely (not
  fully confirmed) cause: LeRobot's backend-detection logic probes
  `import av` at startup even when torchcodec ends up being used, loading
  PyAV's bundled ffmpeg dylib alongside Homebrew's separate ffmpeg 8.1.
  Printed harmlessly both times; revisit only if it ever accompanies an
  actual crash.
- Commit: week3-day2: partition_utils.py + partition_report.py +
  day2_lab_exploration.py -- real LeRobot v3.0 file-group partitioning,
  verified end to end via PyAV (matching the published reference
  architecture) after decord and torchcodec were tried first
- Open questions carried forward:
  - mini_project_ray_vs_spark_timing.py (Day 1) still never run for real.
  - ray_bronze_ingest.py's production-polish version (column pruning,
    expression filter, clean output) still not re-merged into the working file.
  - get_episode_frame_counts_pyav() does one open+seek+decode pass per
    file group (not per-episode) -- fine at this scale, but still runs
    one partition at a time; not yet wired into Ray Data for parallel
    execution across all 4 partitions.
  - pusht stretch goal (does the reduction ratio generalize past aloha's
    3-camera structure?) not run.
  - decord/read_videos() left unfixed on purpose -- available later via a
    conda-pinned ffmpeg 7.x if a future week specifically needs it.
  - duplicate-ffmpeg objc warning not root-caused, just confirmed harmless
    on this run.

### Day 3 — status: complete
- Built: ray_train_stub.py (--run-name/--storage-path resume via same
  (storage_path, name); StateActionCollateFn(NumpyBatchCollateFn) built on
  to_2d_float_array; get_checkpoint() at top with world-size-safe unwrap();
  report() every rank, checkpoint attached rank-0 only; --kill-at-epoch to
  keep the resume path tested; --max-failures for worker-level retry) +
  training_narrative.md (three-way lerobot-train / accelerate launch
  lerobot-train / Ray Train comparison skeleton -- ran-vs-read stated
  explicitly, real numbers not yet filled in) + lab0-lab3 toy scripts
  (worker/PID discovery, 21-row sharding, real-data batch shape,
  save/crash/resume cycle) -- all four labs RUN FOR REAL this session, not
  just syntax-checked.
- Course-material corrections, verified before building: Ray Train V2 is
  default since Ray 2.51.0 (this install's actual V2-specific log
  signatures -- TrainController, RayTrainWorker, WorkerGroupStartupTimeoutError
  -- confirmed present, not just inferred from version number);
  resume_from_checkpoint=/TorchTrainer.restore() deprecated -- V2 resume =
  same RunConfig(storage_path, name); lerobot-train ALREADY does multi-GPU
  DDP via `accelerate launch` (LeRobot v0.4.0) -- Ray Train's real
  differentiators are cluster scheduling, worker fault tolerance
  (FailureConfig), and scaling CPU-decode/GPU-train independently.
- Real gotcha, found only by running it (not anticipated in planning): a
  plain function passed as collate_fn to iter_torch_batches() works but
  prints `RayDeprecationWarning: ... deprecated in Ray 2.47`. Root-caused
  in Ray's own iter_torch_batches source: passing NO collate_fn sets
  batch_format="pyarrow" internally (reads Arrow tensor-extension columns
  directly -- never touches the numpy-object-dtype path that broke Week 3
  Day 1's math); passing your OWN collate_fn flips Ray back to
  batch_format="numpy", where Day 1's object-dtype gotcha reappears --
  confirmed to_2d_float_array is still required inside a custom collate
  function. Fixed by subclassing NumpyBatchCollateFn (ray.data.collate_fn)
  instead of a bare function -- confirmed against Ray's own
  iter_torch_batches API doc example pattern.
- Real numbers confirmed end to end:
  - Lab 1 (sharding): 21 rows / 2 workers -> 10 and 10 (id=20, the 21st
    row, silently dropped -- equal=True doing exactly what it promises).
    Split was interleaved (rank 0 = all even ids, rank 1 = all odd), NOT
    contiguous blocks -- not a documented guarantee, just how this run's
    blocks landed; never rely on WHICH rows land where.
  - Lab 2 (shape): with NO collate_fn, action/observation.state came back
    as real (32, 14) float32 tensors directly -- no fix needed at that
    step (see gotcha above for why). One epoch on real data, world_size=1:
    first_loss=0.573373, last_loss=0.029511 (~19x drop) -- plausible given
    state/action are closely correlated at 50fps teleoperation, not
    necessarily evidence of a sophisticated fit.
  - Lab 3 (checkpoint/resume): run_a killed via os._exit(1) right after
    epoch 2's checkpoint reported (world_size=1) -> trainer.fit()
    correctly raised ray.train.WorkerGroupError/ActorDiedError, because
    FailureConfig defaults to max_failures=0 (no silent retry) -- CONFIRMED
    this large traceback is expected/correct behavior, not a bug. Manual
    rerun of run_a printed "RESUMED at epoch 2" and completed through
    epoch 5. A THIRD invocation of run_a (already fully finished) printed
    "RESUMED at epoch 6", found range(6,6) empty, did zero new work, and
    result.metrics simply echoed epoch 5's old numbers -- a sharper version
    of the "reused run-name" warning than expected: a COMPLETED job rerun
    under the same name looks like a clean success while doing nothing new.
  - Lab 0 (GPU): use_gpu=True timed out after 60s waiting for 2 GPU
    workers -- confirmed Ray never sees this Mac's MPS device as a `GPU`
    resource cluster-wide.
- Commit: week3-day3: ray_train_stub.py (NumpyBatchCollateFn fix folded
  in) + training_narrative.md -- Ray Train V2 mechanics run for real on
  toys (not just syntax-checked), resume tested by an actual kill,
  world-size-safe unwrap, rank-0 checkpointing, honest lerobot-train /
  accelerate / Ray Train comparison
- Open questions carried forward:
  - FailureConfig(max_failures=2) automatic in-process retry never run for
    real (planned: fresh run name run_b, kill at epoch 2, confirm
    trainer.fit() recovers WITHOUT a second manual invocation).
  - Debugging Challenge (num_workers=2, resume the DDP-wrap-only-at-
    world_size>1 checkpoint bug) not yet run -- ray_train_stub.py's
    unwrap() fix is written and reasoned from Ray's V2 source, but not
    personally reproduced via the actual crash-then-fix cycle.
  - Coding Exercise (checkpoint file count at num_workers=2, before/after
    rank-0-only reporting) not yet run.
  - ray_train_stub.py itself not yet run end-to-end against the real
    127,500-row dataset -- confirmed only via py_compile/ast parse plus
    each underlying mechanism individually verified in the labs.
  - training_narrative.md's "what I actually ran" table still has blanks
    -- specifically whether lerobot-train has been run for real in
    robot-policy-lab; fill in before treating the three-way comparison as
    honest rather than aspirational.
  - Still carried from Week 3 Day 1: mini_project_ray_vs_spark_timing.py
    never run; ray_bronze_ingest.py's production-polish version not
    re-merged; get_episode_frame_counts_pyav() not wired into Ray Data for
    parallel execution across partitions -- Day 3's get_dataset_shard is
    the consumer side of exactly that seam, still open.

## Weeks 4–10
(Add a new `## Week N` section as you get to it, same format as above.)