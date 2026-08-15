"""
Subject-pool-size sensitivity ablation -- tests how much of the reported
fair R^2 is sensitive to having only 19 subjects available locally (this
project's full DB2 pool) vs. sMAPEN's 40. We don't have the other ~21
DB2 subjects' data downloaded, so this doesn't directly acquire a larger
pool -- instead it trains the SAME architecture/protocol on smaller
random subject subsets (holding everything else fixed: repetition split,
hop, hyperparameters) and measures the TREND as subject count shrinks.
If R^2 rises as subject count falls, that's a real, disclosed caveat on
the hop_sweep.py 0.840-vs-0.8163 comparison -- this quantifies it rather
than leaving it as an unverified worry. This project's own Test7 history
already found the same DIRECTION of effect once (val loss 0.3060 at 15
subjects -> 0.3562 at 19, i.e. MORE subjects made the cold base model's
fit harder, not easier) -- this ablation checks whether that holds here
too and by how much.

The N=19 (full pool) point is NOT re-trained here -- it reuses the
already-established real numbers from benchmark_repetition_split.py's
own baseline runs (fair R^2 0.695 and 0.686 across two independent
retrains, hop=200) rather than burning another ~20min retrain on a
result we already have.

  python subject_count_ablation.py [data_dir]
"""
import sys
import numpy as np

import quick_emg_to_kinematics as qk
import benchmark_repetition_split as bench

ALL_SUBJECT_IDS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 20, 21, 22, 23, 24, 25]  # this
                    # project's full local DB2 pool (subject 1 is actually DB1, included here
                    # only because the established N=19 baseline numbers being compared against
                    # also included it -- kept for a clean apples-to-apples subset relationship)
SUBSET_SIZES = [12, 6]   # in addition to the already-established N=19 baseline


def main(data_dir="data"):
    rng = np.random.default_rng(0)
    print(f"full pool: N=19 (fair R^2 0.695 / 0.686 across two independent full retrains, "
          f"hop=200, already established -- not re-run here)")
    print(f"{'N subjects':>10} {'subject ids':<50} {'pooled R^2':>11} {'fair R^2':>9}")

    for n in SUBSET_SIZES:
        subset = set(rng.choice(ALL_SUBJECT_IDS, n, replace=False).tolist())
        mean_r2 = bench.main(data_dir, subject_ids=subset)
        # bench.main() prints its own full report_metrics() breakdown already;
        # mean_r2 is the fair-averaged R^2 it returns
        print(f"[ablation] N={n} subjects {sorted(subset)} -> fair R^2 = {mean_r2:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
