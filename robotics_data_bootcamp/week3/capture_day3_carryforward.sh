#!/usr/bin/env bash
# capture_day3_carryforward.sh -- run from robot-data-forge repo root.
# Closes Day 3's three carried-forward items using ray_train_stub.py's
# CONFIRMED flags (from your own --help output):
#   --run-name (required) --root --epochs --batch-size --lr --num-workers
#   --storage-path --max-failures --kill-at-epoch --max-rows --shuffle-buffer

set -u
STORAGE="$HOME/ray_results_day4"
mkdir -p day4_logs

banner () { echo; echo "===== $1 ====="; echo; }

# ---------------------------------------------------------------
banner "D3-1. FailureConfig(max_failures=2) auto-retry -- run_b, kill at epoch 2"
echo "Expect: process hits os._exit(1) after epoch 2's checkpoint, Ray Train's"
echo "controller auto-restarts the worker in-process (max_failures=2 budget),"
echo "resumes from the checkpoint, and finishes through epoch 5 -- ALL in this"
echo "ONE command, no second manual invocation."
python ray_train_stub.py --run-name run_b --num-workers 1 --epochs 5 \
    --max-failures 2 --kill-at-epoch 2 --storage-path "$STORAGE" \
    2>&1 | tee day4_logs/13_day3_failureconfig_run_b.log

# ---------------------------------------------------------------
banner "D3-2a. Debugging Challenge -- num_workers=2, kill at epoch 2"
echo "max-failures NOT set here (defaults to 0) -- this call is EXPECTED to"
echo "raise (WorkerGroupError/ActorDiedError), same as Day 3's world_size=1 test."
python ray_train_stub.py --run-name debug_nw2 --num-workers 2 --epochs 5 \
    --kill-at-epoch 2 --storage-path "$STORAGE" \
    2>&1 | tee day4_logs/14_day3_debug_nw2_kill.log || true

banner "D3-2b. Debugging Challenge -- SAME run-name, manual resume at num_workers=2"
echo "This is the actual test: does resuming a DDP-wrapped (module.-prefixed)"
echo "checkpoint at num_workers=2 load cleanly, or does it crash on a key"
echo "mismatch? Your unwrap() fix should make this complete without error."
python ray_train_stub.py --run-name debug_nw2 --num-workers 2 --epochs 5 \
    --storage-path "$STORAGE" \
    2>&1 | tee day4_logs/15_day3_debug_nw2_resume.log

# ---------------------------------------------------------------
banner "D3-3. Coding Exercise -- checkpoint file count at num_workers=2"
python ray_train_stub.py --run-name coding_ex_nw2 --num-workers 2 --epochs 3 \
    --storage-path "$STORAGE" \
    2>&1 | tee day4_logs/16_day3_coding_ex_nw2.log

echo "--- checkpoint files actually written (expect ONE per epoch, not two) ---"
ls -R "$STORAGE/coding_ex_nw2" 2>&1 | tee day4_logs/17_day3_checkpoint_listing.log

banner "DONE. Paste back day4_logs/13-17."
