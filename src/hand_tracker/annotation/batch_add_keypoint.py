#!/usr/bin/env python3
"""
batch_add_keypoint.py

Add a new keypoint to every JARVIS-style annotations.csv found under a root
directory, e.g.:

    root/
      trial1/camBL/annotations.csv
      trial1/camBR/annotations.csv
      trial2/camBL/annotations.csv
      trial2/camBR/annotations.csv

By default it looks for files matching "*/*/annotations.csv" (trial/cam
folders), but you can pass a different --pattern for other layouts (glob
syntax, relative to --root).

Usage:
    python batch_add_keypoint.py ROOT --name Arm [--pattern "*/*/annotations.csv"]
                                  [--entity HandObject] [--after Wrist_R]
                                  [--state 0] [--in-place | --out-dir OUT_ROOT]
                                  [--suffix _with_arm]

Output modes (pick one):
    --in-place            Overwrite each annotations.csv directly (a .bak
                           backup of the original is kept alongside it).
    --out-dir OUT_ROOT     Mirror the trial/cam folder structure under
                           OUT_ROOT instead of touching the originals.
    (default, if neither given): write alongside each original file with
                           --suffix inserted before .csv, e.g.
                           annotations_with_arm.csv

Examples:
    # Dry run mirrored into a new folder, keeping originals untouched
    python batch_add_keypoint.py ./data --name Arm --out-dir ./data_with_arm

    # Update files in place (keeps annotations.csv.bak backups)
    python batch_add_keypoint.py ./data --name Arm --in-place
"""

import argparse
import shutil
from pathlib import Path

from add_keypoint import add_keypoint_to_csv, KeypointExistsError


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Root directory containing the trial/cam folders")
    ap.add_argument("--name", required=True, help="Name of the new keypoint to add")
    ap.add_argument("--pattern", default="*/*/annotations.csv",
                    help="Glob pattern (relative to root) to find CSVs. Default: */*/annotations.csv")
    ap.add_argument("--entity", default=None, help="Entity/individual label for the new keypoint")
    ap.add_argument("--after", default=None, help="Existing bodypart to insert the new keypoint after")
    ap.add_argument("--state", default="0", help="Default state value for existing rows (default: 0)")

    out_group = ap.add_mutually_exclusive_group()
    out_group.add_argument("--in-place", action="store_true", help="Overwrite files in place (keeps a .bak backup)")
    out_group.add_argument("--out-dir", default=None, help="Mirror folder structure under this output root instead")
    ap.add_argument("--suffix", default="_with_new_keypoint",
                    help="Suffix for output filenames when not using --in-place/--out-dir "
                         "(default: _with_new_keypoint)")

    args = ap.parse_args()

    root = Path(args.root)
    csv_paths = sorted(root.glob(args.pattern))

    if not csv_paths:
        raise SystemExit(f"No files matched pattern '{args.pattern}' under {root}")

    print(f"Found {len(csv_paths)} file(s) matching '{args.pattern}' under {root}\n")

    n_ok, n_skipped, n_failed = 0, 0, 0

    for src in csv_paths:
        rel = src.relative_to(root)

        if args.in_place:
            dest = src
            backup = src.with_suffix(src.suffix + ".bak")
            shutil.copy2(src, backup)
            read_from = backup  # read from the untouched backup, write to the original path
        elif args.out_dir:
            dest = Path(args.out_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            read_from = src
        else:
            dest = src.with_name(f"{src.stem}{args.suffix}{src.suffix}")
            read_from = src

        try:
            entity_value, n_frames = add_keypoint_to_csv(
                str(read_from), str(dest),
                name=args.name, entity=args.entity, after=args.after, state=args.state,
            )
            print(f"[ok]      {rel}  ->  {dest}  ({n_frames} frames, entity='{entity_value}')")
            n_ok += 1
        except KeypointExistsError as e:
            print(f"[skipped] {rel}: {e}")
            n_skipped += 1
        except ValueError as e:
            print(f"[failed]  {rel}: {e}")
            n_failed += 1

    print(f"\nDone: {n_ok} updated, {n_skipped} skipped (already had '{args.name}'), {n_failed} failed.")


if __name__ == "__main__":
    main()
