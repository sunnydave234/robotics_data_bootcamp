# Spark vs. Ray Data: Engine Boundary, Week 3

## Coding Exercise answers (episode_quality)

Real numbers from `coding_exercise_episode_quality.py` / `spark_tabular_warehouse.py --ray-output`:
- episodes total: 85
- episodes with sync_error_flag: 0
- episodes with n_outlier_frames > 0: 0
- max outlier_frame_rate: 0.0
- episodes with ANY flag (unified): 0
- (context: ray_bronze_ingest_simplified.py's dataset-wide stats: mean=2.1567, std=0.7756, 0/127500 frames flagged at 3σ)

1. How many episodes have `n_outlier_frames > 0`? What's the max `outlier_frame_rate`?
   > _[your answer -- reference the numbers above, and say what you think it means that this dataset came back at zero]_

2. Where did unification happen, and why there?
   > _[your answer]_

3. What does the unified table cost you that two separate signals don't?
   > _[your answer]_

4. What does keeping them separate cost?
   > _[your answer]_

## Engine choice, one row per Week 3 artifact

| Artifact (Days 1–4) | Engine used | Touches a pixel or a GPU? | Was that the right engine? Why, in one sentence |
|---|---|---|---|
| `ray_bronze_ingest_simplified.py` — action-magnitude outlier flag | Ray Data | **No** — `np.linalg.norm` on a 14-float column | _[your answer]_ |
| `partition_utils.py` + PyAV decode across file groups | Ray Data (custom, per Anyscale's reference) | Yes — decodes frames | _[your answer]_ |
| `ray_train_stub.py` — DDP + checkpoint/resume | Ray Train | Yes — GPU (would be, off this Mac) | _[your answer]_ |
| `episodes` / `robots` / `sync` | Spark | No | _[your answer]_ |
| `episode_quality` (frame → episode rollup + join) | Spark, reading Ray's output | No | _[your answer -- also say where you'd move it if the tables were 100x smaller, or 100x larger]_ |
| `mini_project_ray_vs_spark_timing.py` | both | No | _[your answer, once you've actually run the Stretch Goal]_ |

## Note on ray_train_stub.py's checkpoint-resume bug (Day 3 carry-forward)

Confirmed via `checkpoint_manager_snapshot.json` and the installed Ray 2.58.0 source
(`checkpoint_manager.py`, `report_handler.py`, `controller.py:436`): a hard kill
(`os._exit(1)`) placed immediately after `ray.train.report()` can deterministically
lose that exact report. The controller retrieves reports by polling the worker group
on its own async cycle (`poll_status(timeout=health_check_interval_s)`) rather than
receiving them synchronously when `report()` returns -- so a kill that lands before
the next poll cycle means that epoch's checkpoint was written to disk correctly but
never reaches `register_checkpoint()`, and `get_checkpoint()` on resume falls back to
the previous epoch's checkpoint instead. Reproduced identically across 5 separate runs
(run_b through run_f); a 1-second `time.sleep()` in the worker had zero effect, since
the gap is cross-process and not fixable from training-loop code.

Practical implication: `FailureConfig`'s retry budget can be silently consumed
re-executing an epoch that already completed and already checkpointed.
