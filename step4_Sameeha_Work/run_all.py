"""
run_all.py -> runs all my Task 2 files in order and says PASS or FAIL for each

    python step4_Sameeha_Work/run_all.py

Runs step 1-7, observer.py and step8_testing.py (takes about 1 minute). The graphs get saved
in results/ instead of popping up. The last line is step 8's result.
"""

# os is imported to pass settings (environment variables) to each file
import os
# subprocess is imported to run each file, just like pressing the play button on it
import subprocess
import sys
# time is imported to show how long each file takes
import time
from pathlib import Path

# my folder
HERE = Path(__file__).resolve().parent
# the files to run, in order
FILES = ["step1_state_space_model.py", "step2_controllability_observability.py",
         "step3_measurement_and_ground_truth.py", "step4_observers.py", "step5_evaluation.py",
         "step6_tuning.py", "step7_robustness.py", "observer.py", "step8_testing.py"]

if __name__ == "__main__":
    # MPLBACKEND=Agg saves the graphs without popping up windows, so it doesn't stop and wait for me
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    failed, last_out = [], ""
    # running each file one after another
    for f in FILES:
        t0 = time.time()
        # running the file and catching everything it prints
        r = subprocess.run([sys.executable, str(HERE / f)], capture_output=True, text=True, env=env)
        # returncode 0 = it worked, anything else = it crashed (or step 8 had a FAIL)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {f:42s} {time.time() - t0:6.1f} s")
        # if it failed, print the end of the error so I can see what went wrong
        if not ok:
            failed.append(f)
            print("      " + (r.stderr or r.stdout).strip()[-1500:].replace("\n", "\n      "))
        # keeping what the last file (step 8) printed
        last_out = r.stdout
    # finding step 8's 'RESULT: 24 / 24' line to print at the end
    result = [l for l in last_out.splitlines() if l.startswith("RESULT")]
    print("\n" + (result[0] if result else "step 8 did not report a result"))
    print("ALL FILES RAN - graphs and report in step4_Sameeha_Work/results/" if not failed
          else f"FAILED: {failed}")
    # exit code 1 if anything failed
    sys.exit(1 if failed else 0)