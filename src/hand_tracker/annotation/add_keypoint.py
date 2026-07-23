#!/usr/bin/env python3
"""
add_keypoint.py

Add a new keypoint (bodypart) to a JARVIS/DeepLabCut-style annotation CSV.

These CSVs have a 4-row header:
    Row 1: Scorer      (repeated for every column)
    Row 2: entities     (e.g. HandObject, repeated for every column)
    Row 3: bodyparts   (the keypoint name, repeated 3x per keypoint)
    Row 4: coords      (x, y, state -- repeated per keypoint)
Followed by one data row per labeled frame.

This script inserts 3 new columns (x, y, state) for a brand-new keypoint,
filling all existing frames with blank/unlabeled values (,,0) so the file
stays valid and can be opened in JARVIS/DLC for labeling the new point.

Usage:
    python add_keypoint.py INPUT.csv OUTPUT.csv --name Arm [--entity HandObject] [--after Wrist_R] [--state 0]

Options:
    --name    Name of the new keypoint/bodypart to add (required)
    --entity  Value to use in the "entities" row for the new keypoint.
              Defaults to the entity of the last existing keypoint.
    --after   Name of an existing bodypart to insert the new keypoint after.
              If omitted, the new keypoint is appended at the very end.
    --state   Default "state" value to fill for existing (already labeled)
              frames. JARVIS uses 0 = not visible/unlabeled. Default: 0
"""

import csv
import argparse


class KeypointExistsError(Exception):
    """Raised when the requested keypoint already exists in the file."""


def add_keypoint_to_csv(input_csv, output_csv, name, entity=None, after=None, state="0"):
    """
    Add a new keypoint (3 columns: x, y, state) to a JARVIS/DLC-style
    annotation CSV. Returns (entity_value, n_frames) on success.

    Raises KeypointExistsError if `name` is already a bodypart in the file,
    and ValueError for other structural problems (bad --after, too few rows).
    """
    with open(input_csv, newline="") as f:
        rows = [row for row in csv.reader(f)]

    if len(rows) < 4:
        raise ValueError("Expected at least 4 header rows (Scorer/entities/bodyparts/coords).")

    scorer_row, entity_row, bodypart_row, coords_row = rows[0], rows[1], rows[2], rows[3]
    data_rows = rows[4:]

    n_header_cols = len(scorer_row)

    # Some JARVIS exports have a trailing comma on data rows (one extra blank
    # field vs. the header rows). Trim that artifact off before we work with
    # the data so column counts line up; we won't re-add it in the output.
    trimmed_data_rows = []
    for row in data_rows:
        if len(row) == n_header_cols + 1 and row[-1] == "":
            row = row[:-1]
        trimmed_data_rows.append(row)

    # Figure out where each keypoint's 3 columns (x,y,state) start.
    # Column 0 is the frame-name/label column; keypoints start at column 1.
    bodyparts_in_order = []
    seen = set()
    for i in range(1, len(bodypart_row), 3):
        bp = bodypart_row[i]
        if bp not in seen:
            bodyparts_in_order.append(bp)
            seen.add(bp)

    if name in bodyparts_in_order:
        raise KeypointExistsError(f"Keypoint '{name}' already exists in this file.")

    # Determine insertion index (in terms of "column index", 1-based groups of 3)
    if after:
        if after not in bodyparts_in_order:
            raise ValueError(f"--after '{after}' not found. Existing keypoints: {bodyparts_in_order}")
        insert_after_idx = bodyparts_in_order.index(after)
        insert_col = 1 + (insert_after_idx + 1) * 3  # column right after that keypoint's 3 cols
    else:
        insert_col = len(scorer_row)  # append at the very end

    entity_value = entity if entity else entity_row[-1]

    def insert_triplet(row, col, values):
        return row[:col] + list(values) + row[col:]

    new_scorer_row = insert_triplet(scorer_row, insert_col, [scorer_row[-1]] * 3)
    new_entity_row = insert_triplet(entity_row, insert_col, [entity_value] * 3)
    new_bodypart_row = insert_triplet(bodypart_row, insert_col, [name] * 3)
    new_coords_row = insert_triplet(coords_row, insert_col, ["x", "y", "state"])

    new_data_rows = [insert_triplet(row, insert_col, ["", "", state]) for row in trimmed_data_rows]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_scorer_row)
        writer.writerow(new_entity_row)
        writer.writerow(new_bodypart_row)
        writer.writerow(new_coords_row)
        writer.writerows(new_data_rows)

    return entity_value, len(new_data_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--name", required=True, help="Name of the new keypoint to add")
    ap.add_argument("--entity", default=None, help="Entity/individual label for the new keypoint")
    ap.add_argument("--after", default=None, help="Existing bodypart to insert the new keypoint after (default: append at end)")
    ap.add_argument("--state", default="0", help="Default state value for existing rows (default: 0)")
    args = ap.parse_args()

    try:
        entity_value, n_frames = add_keypoint_to_csv(
            args.input_csv, args.output_csv,
            name=args.name, entity=args.entity, after=args.after, state=args.state,
        )
    except (KeypointExistsError, ValueError) as e:
        raise SystemExit(str(e))

    print(f"Added keypoint '{args.name}' (entity='{entity_value}') to {n_frames} frames.")
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
