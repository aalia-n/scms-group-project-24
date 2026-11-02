"""
run_all.py -> runs all my Task 2 files in order and says PASS or FAIL for each

    python step4_Sameeha_Work/run_all.py

Runs step 1-7, observer.py and step8_testing.py (takes about 1 minute). The graphs get saved
in results/ instead of popping up. The last line is step 8's result.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["step1_state_space_model.py", "step2_controllability_observability.py",
         "step3_measurement_and_ground_truth.py", "step4_observers.py", "step5_evaluation.py",
         "step6_tuning.py", "step7_robustness.py", "observer.py", "step8_testing.py"]

if __name__ == "__main__":
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    failed, last_out = [], ""
    for f in FILES:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / f)], capture_output=True, text=True, env=env)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {f:42s} {time.time() - t0:6.1f} s")
        if not ok:
            failed.append(f)
            print("      " + (r.stderr or r.stdout).strip()[-1500:].replace("\n", "\n      "))
        last_out = r.stdout
    result = [l for l in last_out.splitlines() if l.startswith("RESULT")]
    print("\n" + (result[0] if result else "step 8 did not report a result"))
    print("ALL FILES RAN - graphs and report in step4_Sameeha_Work/results/" if not failed
          else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)
