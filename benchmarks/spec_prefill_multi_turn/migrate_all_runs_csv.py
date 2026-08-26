#!/usr/bin/env python3
"""One-time migration: add the `scbench_config` column to an existing
`results/all_runs.csv`.

`predict_scbench.py` appends to `all_runs.csv` and writes the header only
when the file does not yet exist (`ensure_csv_header`). So adding a field to
`CSV_FIELDS` makes every new row carry one more value than the old header
declares -- the file silently goes ragged, and `csv.DictReader` starts
folding the extra value under a `None` key. This script rewrites the file
under the new header before that happens.

**What it does NOT do: guess.** Pre-existing rows get an empty
`scbench_config`, because the value is not recoverable from the row. It is
*inferable* -- `num_conversations` is 23 for `scbench_qa_eng`, ~70 for
`scbench_summary`, ~100 for `scbench_kv`, ~193 for an unrestricted run --
but summary and qa_eng are close enough on the loaded count that a wrong
guess is plausible, and a wrong value written into a results file is worse
than a blank one. Pass `--infer` to fill them in anyway from
`--conversation-counts`, which prints what it inferred so you can check it.

Usage:
    python3 migrate_all_runs_csv.py                     # blanks, safest
    python3 migrate_all_runs_csv.py --infer             # fill from counts
    python3 migrate_all_runs_csv.py --dry-run           # report only
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from predict_scbench import CSV_FIELDS  # noqa: E402

DEFAULT_CSV = Path(os.environ.get("BENCH_RESULTS_DIR", "results")) / "all_runs.csv"
NEW_COLUMN = "scbench_config"

# num_conversations -> config, for --infer. Exact matches only; anything
# else is left blank rather than snapped to the nearest bucket.
DEFAULT_COUNTS = {
    23: "scbench_qa_eng",
    70: "scbench_summary",
    100: "scbench_kv",
    193: "scbench_kv+scbench_qa_eng+scbench_summary",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--infer", action="store_true",
                        help="Fill pre-existing rows from --conversation-counts "
                             "instead of leaving them blank.")
    parser.add_argument("--conversation-counts", default=None,
                        help="Override the inference table, e.g. "
                             "'23=scbench_qa_eng,70=scbench_summary,100=scbench_kv'.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"{args.csv} does not exist -- nothing to migrate")

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        rows = list(reader)

    if NEW_COLUMN in old_fields:
        print(f"[migrate] {args.csv} already has '{NEW_COLUMN}' -- nothing to do")
        return

    counts = dict(DEFAULT_COUNTS)
    if args.conversation_counts:
        counts = {}
        for pair in args.conversation_counts.split(","):
            n, name = pair.split("=", 1)
            counts[int(n)] = name.strip()

    dropped = [c for c in old_fields if c not in CSV_FIELDS]
    if dropped:
        parser.error(
            f"{args.csv} has column(s) the current CSV_FIELDS does not: {dropped}. "
            f"That means the schema moved in some other way too -- resolve that "
            f"before migrating, or this rewrite would silently drop those values."
        )

    filled = 0
    for row in rows:
        # DictReader folds any surplus values under None; a ragged file is
        # exactly the failure this migration exists to prevent, so say so.
        if None in row:
            parser.error(
                f"{args.csv} is already ragged (a row has more values than the "
                f"header declares). Restore it from a backup before migrating."
            )
        value = ""
        if args.infer:
            try:
                value = counts.get(int(row.get("num_conversations") or 0), "")
            except ValueError:
                value = ""
            if value:
                filled += 1
        row[NEW_COLUMN] = value

    print(f"[migrate] {args.csv}: {len(rows)} row(s), {len(old_fields)} -> "
          f"{len(CSV_FIELDS)} columns"
          + (f", {filled} inferred, {len(rows) - filled} blank" if args.infer
             else f", {len(rows)} blank"))
    if args.infer:
        seen = sorted({r[NEW_COLUMN] for r in rows if r[NEW_COLUMN]})
        for v in seen:
            print(f"           inferred {sum(1 for r in rows if r[NEW_COLUMN] == v):3d} x {v}")

    if args.dry_run:
        print("[migrate] --dry-run: nothing written")
        return

    backup = args.csv.with_suffix(".csv.pre-config-column.bak")
    shutil.copy2(args.csv, backup)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    print(f"[migrate] backed up -> {backup}")
    print(f"[migrate] rewrote   -> {args.csv}")


if __name__ == "__main__":
    main()
