"""Non-linear (calibrated) multi-view EKS for the litpose_multiview dataset.

For each session in ``video_preds/`` of the chosen model directory this script:

1. Loads the 5 per-view raw prediction CSVs.
2. Finds the matching per-session calibration TOML in ``<data_dir>/calibrations/``.
3. Runs the non-linear (camgroup-projected) multi-camera EKS smoother.
4. Saves per-view smoothed CSVs (plus the 3D latent CSV) to
   ``<model_dir>/non_linear_eks/video_preds/``.

The model directory holds a SINGLE trained model (no ensemble of seeds), so EKS
runs with ``n_models=1``. In that case the EKS ensemble step replaces the (zero)
sample variance with ``1/max(confidence, 0.05)`` per keypoint/frame so the
filter behaves well.

Sessions whose calibration TOML is missing are skipped (and logged) since the
user explicitly asked for non-linear EKS only.

Usage:
    python lp3d-analysis/scripts/run_nonlinear_eks_litpose_multiview.py \
        [--model-dir ...] [--data-dir ...] [--overwrite]
"""

from __future__ import annotations

import os

# Set BEFORE importing JAX-using libs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import argparse
import re
import sys
import traceback
from pathlib import Path

import yaml

# Make sure the local lp3d-analysis package is importable when running as a script.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eks.utils import format_data  # noqa: E402
from eks.marker_array import input_dfs_to_markerArray  # noqa: E402
from eks.multicam_smoother import ensemble_kalman_smoother_multicam  # noqa: E402


DEFAULT_MODEL_DIR = "/teamspace/studios/this_studio/litpose_multiview/models/reproject_weight_3"
DEFAULT_DATA_DIR = "/teamspace/studios/this_studio/litpose_multiview"


def discover_sessions(video_preds_dir: Path, views: list[str]) -> dict[str, dict[str, Path]]:
    """Return {session_name: {view: csv_path}} for sessions with all views present.

    Sessions are inferred from prediction file names of the form
    ``<session>_<view>.csv``. ``*_temporal_norm.csv`` files are ignored.
    """
    sorted_views = sorted(views, key=len, reverse=True)
    view_re = "|".join(re.escape(v) for v in sorted_views)
    pat = re.compile(rf"^(?P<session>.+)_(?P<view>{view_re})\.csv$")

    sessions: dict[str, dict[str, Path]] = {}
    for f in sorted(video_preds_dir.iterdir()):
        if not f.is_file() or not f.name.endswith(".csv"):
            continue
        if f.name.endswith("_temporal_norm.csv"):
            continue
        if "_pixel_error" in f.name:
            continue
        m = pat.match(f.name)
        if m is None:
            continue
        sessions.setdefault(m.group("session"), {})[m.group("view")] = f

    complete = {s: v for s, v in sessions.items() if all(view in v for view in views)}
    incomplete = {s: v for s, v in sessions.items() if not all(view in v for view in views)}
    if incomplete:
        print(
            f"  [warn] {len(incomplete)} sessions are missing one or more views and will be skipped:"
        )
        for s, v in incomplete.items():
            print(f"    {s}: have {sorted(v)}, missing {sorted(set(views) - set(v))}")
    return complete


def load_views_from_cfg(cfg_path: Path) -> list[str]:
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    views = list(cfg["data"]["view_names"])
    return views


def run_session(
    session: str,
    view_to_csv: dict[str, Path],
    views: list[str],
    calibration_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> bool:
    """Run non-linear EKS for one session. Returns True on success, False on skip/failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {v: output_dir / f"{session}_{v}.csv" for v in views}
    out_3d = output_dir / f"{session}_3d.csv"

    if not overwrite and all(p.exists() for p in out_paths.values()) and out_3d.exists():
        print(f"  [skip] {session} - outputs already exist")
        return False

    csv_paths = [str(view_to_csv[v]) for v in views]
    print(f"  Loading {len(csv_paths)} CSVs (one per view) for {session}")
    markers_list, keypoint_names = format_data(
        input_source=csv_paths,
        camera_names=views,
    )

    min_frames = min(len(df) for view_dfs in markers_list for df in view_dfs)
    markers_list = [[df.iloc[:min_frames] for df in view_dfs] for view_dfs in markers_list]
    print(f"  {min_frames} frames, {len(keypoint_names)} keypoints")
    print(f"  Calibration: {calibration_path.name}")

    # Load calibration for non-linear EKS
    from aniposelib.cameras import CameraGroup

    camgroup = CameraGroup.load(str(calibration_path))

    # Build marker array and call the multicam smoother directly so we can pass
    # an ``s_frames`` window sized to the actual session length.
    marker_array = input_dfs_to_markerArray(markers_list, keypoint_names, camera_names=views)
    # Window of frames used to optimize the smoothing parameter only.
    # 1000 matches the upstream default in run_eks_multiview; the final
    # smoothing pass always runs over all frames.
    s_window_end = min(min_frames, 1000)
    s_frames = [(0, s_window_end)]
    print(f"  Using s_frames={s_frames} for smooth-param optimization")

    camera_dfs, _smooth_params, df_3d = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=None,
        quantile_keep_pca=50,
        camera_names=views,
        s_frames=s_frames,
        avg_mode="median",
        var_mode="confidence_weighted_var",
        inflate_vars=True,
        inflate_vars_kwargs={},
        verbose=True,
        pca_object=None,
        n_latent=3,
        camgroup=camgroup,
    )

    for view_idx, v in enumerate(views):
        if view_idx >= len(camera_dfs) or camera_dfs[view_idx] is None:
            print(f"  [warn] no result for view {v}")
            continue
        camera_dfs[view_idx].to_csv(out_paths[v])
        print(f"  saved {out_paths[v].name}")

    if df_3d is not None:
        df_3d.to_csv(out_3d)
        print(f"  saved {out_3d.name}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--video-preds-subdir", default="video_preds")
    parser.add_argument("--output-subdir", default="non_linear_eks/video_preds")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional list of session names to restrict to.",
    )
    args = parser.parse_args()

    cfg_path = args.model_dir / "config.yaml"
    views = load_views_from_cfg(cfg_path)
    print(f"Views from config: {views}")

    video_preds_dir = args.model_dir / args.video_preds_subdir
    calib_dir = args.data_dir / "calibrations"
    out_dir = args.model_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model dir:       {args.model_dir}")
    print(f"Video preds dir: {video_preds_dir}")
    print(f"Calibration dir: {calib_dir}")
    print(f"Output dir:      {out_dir}")

    sessions = discover_sessions(video_preds_dir, views)
    if args.sessions:
        sessions = {s: v for s, v in sessions.items() if s in args.sessions}
    print(f"Discovered {len(sessions)} complete sessions with all {len(views)} views")

    sessions_no_calib: list[str] = []
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for i, (session, view_to_csv) in enumerate(sorted(sessions.items()), start=1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(sessions)}] {session}")

        calib_path = calib_dir / f"{session}.toml"
        if not calib_path.exists():
            print(f"  [skip] calibration TOML not found: {calib_path}")
            sessions_no_calib.append(session)
            continue

        try:
            run_session(
                session=session,
                view_to_csv=view_to_csv,
                views=views,
                calibration_path=calib_path,
                output_dir=out_dir,
                overwrite=args.overwrite,
            )
            successes.append(session)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {session}: {e}")
            traceback.print_exc()
            failures.append((session, str(e)))

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  ok:                {len(successes)}")
    print(f"  no calibration:    {len(sessions_no_calib)}")
    print(f"  failed:            {len(failures)}")
    if sessions_no_calib:
        print("  sessions without calibration (skipped):")
        for s in sessions_no_calib:
            print(f"    - {s}")
    if failures:
        print("  failed sessions:")
        for s, msg in failures:
            print(f"    - {s}: {msg}")
    print(f"\nOutputs in: {out_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
