"""Headless run of the dashboard's training pipeline, meant for a Kaggle notebook.

In a Kaggle cell:
    !pip install -q -r requirements.txt
    !python kaggle_run.py --data-dir /kaggle/input/<cert-dataset> \
                          --answers /kaggle/input/<cert-dataset>/answers/insiders.csv

It writes everything to /kaggle/working/insider_out and zips the small files you need
(results, scores, model, feature cache) into /kaggle/working/insider_run.zip. Download that zip,
unzip it on your computer, and open it in the dashboard's "Load Kaggle results" tab.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from app.pipeline import train_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder holding (or above) logon.csv, device.csv, ...")
    ap.add_argument("--answers", required=True, help="insiders.csv / answers.csv")
    ap.add_argument("--out", default="/kaggle/working/insider_out")
    ap.add_argument("--release", default="r4.2")
    ap.add_argument("--window-days", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=7)
    ap.add_argument("--internal-domain", default="@dtaa.com")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--max-benign", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--sampler", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-budget", type=float, default=0.01)
    ap.add_argument("--user-budget", type=float, default=0.10)
    a = ap.parse_args()

    config = {
        "data_dir": a.data_dir, "answers": a.answers, "run_dir": a.out, "release": a.release,
        "window_days": a.window_days, "seq_len": a.seq_len, "internal_domain": a.internal_domain,
        "chunksize": a.chunksize, "max_benign": a.max_benign, "rebuild": a.rebuild,
        "sampler": a.sampler, "seed": a.seed, "window_budget": a.window_budget,
        "user_budget": a.user_budget,
    }
    result = train_run(config, lambda msg: print(msg, flush=True))
    sel = result["metrics"]["user_selected"]
    um = result["metrics"]["user"][sel]["test"]
    print(f"\nHeld-out users: PR-AUC {um['pr_auc']:.3f}, recall {um['recall']:.1%}, "
          f"precision {um['precision']:.1%}, flagged {um['n_flagged']}")

    out = Path(a.out)
    zip_path = out.parent / "insider_run.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [*out.glob("features_w*.pkl"), out / "results.json", out / "test_user_scores.csv",
                  *(out / "model").glob("*")]:
            if f.is_file():
                z.write(f, f.relative_to(out))
    print(f"\nDownload: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
