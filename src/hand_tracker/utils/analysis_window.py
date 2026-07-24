
from pathlib import Path
import pandas as pd

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")

def load_window_lookup(session_name):
    """Load {trial_name: (min_hold_start_frame, min_hold_end_frame)} for a session,
    as produced by get_min_holding_window.py."""
    window_csv_path = ANALYSIS_ROOT / session_name / "log" / "min_holding_window.csv"
    if not window_csv_path.exists():
        return None

    window_df = pd.read_csv(window_csv_path)
    return {
        row["trial_name"]: (row["min_hold_start_frame"], row["min_hold_end_frame"])
        for _, row in window_df.iterrows()
    }