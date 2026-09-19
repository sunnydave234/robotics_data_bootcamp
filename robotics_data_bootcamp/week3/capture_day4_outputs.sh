#!/usr/bin/env bash
# capture_day4_outputs.sh -- run from robot-data-forge repo root.
# Runs every Day 4 lab/exercise/script and tees output into day4_logs/.
# Some commands are EXPECTED to fail (that's the lesson) -- '|| true'
# on those so the script keeps going and still captures the traceback.

set -u
ROOT="$HOME/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
REPO_ID="lerobot/aloha_mobile_cabinet"
mkdir -p day4_logs

banner () { echo; echo "===== $1 ====="; echo; }

# ---------------------------------------------------------------
banner "0a. Environment: Java + PySpark versions"
java -version 2>&1 | tee day4_logs/00_java_version.log
python -c "import pyspark; print('pyspark', pyspark.__version__)" 2>&1 | tee day4_logs/00_pyspark_version.log

banner "0b. Lab 0 -- Spark smoke test (timed)"
{ time python lab0_spark_smoke.py ; } > day4_logs/01_lab0.log 2>&1
cat day4_logs/01_lab0.log

# ---------------------------------------------------------------
banner "1. Day 1 carry-forward -- run ray_bronze_ingest_simplified.py FOR REAL"
python ray_bronze_ingest_simplified.py --repo-id "$REPO_ID" 2>&1 | tee day4_logs/02_ray_bronze_ingest.log
ls -la ray_output_day1/ 2>&1 | tee day4_logs/02_ray_output_listing.log

# ---------------------------------------------------------------
banner "2a. Lab 1 Step A -- tilde-path repro (EXPECTED to fail)"
python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').appName('tilde-repro').getOrCreate()
try:
    spark.read.parquet('~/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet/data/').count()
except Exception as e:
    print('EXPECTED FAILURE:', repr(e))
spark.stop()
" 2>&1 | tee day4_logs/03_lab1_stepA_tilde_error.log || true

banner "2b. Lab 1 Step C -- dotted-column repro (EXPECTED to fail)"
python3 -c "
from pathlib import Path
from pyspark.sql import SparkSession
ROOT = Path.home() / '.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet'
spark = SparkSession.builder.master('local[*]').appName('dotted-repro').getOrCreate()
raw = spark.read.parquet(str(ROOT / 'data' / '*' / '*.parquet'))
raw.printSchema()
try:
    raw.select('observation.state').show(2)
except Exception as e:
    print('EXPECTED FAILURE:', repr(e))
spark.stop()
" 2>&1 | tee day4_logs/04_lab1_stepC_dotted_error.log || true

banner "2c. Lab 1 -- your fixed lab1_episodes.py (schema + reconcile)"
python lab1_episodes.py 2>&1 | tee day4_logs/05_lab1_final.log

# ---------------------------------------------------------------
banner "3. Lab 2 -- sync table (toy + real)"
python lab2_sync.py 2>&1 | tee day4_logs/06_lab2_sync.log

# ---------------------------------------------------------------
banner "4a. Mini Project run #1 -- tabular-only (episodes/sync/robots)"
python spark_tabular_warehouse.py --root "$ROOT" --out spark_output 2>&1 | tee day4_logs/07_warehouse_run1.log

banner "4b. Lab 3 -- inspect the partition trap on spark_output/episodes/"
ls -R spark_output/episodes/ 2>&1 | tee day4_logs/08_lab3_ls.log
python3 -c "
import pyarrow.parquet as pq, glob
f = glob.glob('spark_output/episodes/robot=*/*.parquet')[0]
print('inspecting file:', f)
print(pq.read_schema(f))
" 2>&1 | tee day4_logs/09_lab3_schema.log

# ---------------------------------------------------------------
banner "5. Debugging Challenge -- three-reader dtype comparison"
python debug_challenge_dtypes.py --spark-episodes spark_output/episodes --ray-output ray_output_day1 \
    2>&1 | tee day4_logs/10_debug_challenge_dtypes.log

# ---------------------------------------------------------------
banner "6. Coding Exercise -- episode_quality rollup"
python coding_exercise_episode_quality.py --root "$ROOT" --ray-output ray_output_day1 \
    2>&1 | tee day4_logs/11_coding_exercise.log

# ---------------------------------------------------------------
banner "7. Mini Project run #2 -- full warehouse WITH episode_quality"
python spark_tabular_warehouse.py --root "$ROOT" --out spark_output --ray-output ray_output_day1 \
    2>&1 | tee day4_logs/12_warehouse_run2.log

# ---------------------------------------------------------------
banner "DONE. Everything is under day4_logs/. Paste those files back, plus:"
echo "  - spark_vs_ray_boundary.md (your filled-in table)"
echo "  - git log -1 --stat   (your actual commit)"
echo "  - Day 3 carry-forward outputs -- see separate note, flags unconfirmed against your real file"
