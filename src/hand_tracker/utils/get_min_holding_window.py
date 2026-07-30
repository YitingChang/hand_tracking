
# Get the minimal holding time (frame number)
import os
from pathlib import Path
from glob import glob
import json
import numpy as np
import pandas as pd
from hand_tracker.utils.file_io import load_json, get_trialnames, find_log_or_robot
from read_state_data import read_data_from_txt_file

# --- CONFIGURATION ---
DATA_ROOT = Path("/media/yiting/NewVolume/Data")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
FPS = 100.0  # hz

def process_trial(log_fname, robot_path):
    ####################
    ### Load data ###
    ####################

    # Read compressed robot_state_data file 
    df, timestamps, packet_loss_times, state_dict_list = read_data_from_txt_file(robot_path)
    # df is a dataframe with all robot data from that trial. It should correctly handle dropped packets
    trial_start_robot_timestamp_us = df["CPU_steady_clock_time_us"].iloc[0].item()

    # Read log file and extract halt_motion timestamps, wait_start timestamps
    log_data = load_json(log_fname)
    # log_halt_motion_start_robot_timestamps_us = np.array(log_data["halt_motion_start_robot_timestamps_us"])
    # log_halt_motion_end_robot_timestamps_us = np.array(log_data["halt_motion_end_robot_timestamps_us"])
    # log_wait_start_robot_timestamp_us = np.array(log_data["wait_start_robot_timestamp_us"])


    # Safely get timestamps and handle potential None values
    log_halt_motion_start_robot_timestamps_us = log_data.get("halt_motion_start_robot_timestamps_us")
    log_wait_start_robot_timestamp_us = log_data.get("wait_start_robot_timestamp_us")
    log_halt_motion_end_robot_timestamps_us = log_data.get("halt_motion_end_robot_timestamps_us", [])

    if log_wait_start_robot_timestamp_us is None:
        print(f"Warning: Missing 'wait_start_robot_timestamp_us' in {log_fname}. Skipping trial.")
        return None, None, None, None


    ####################
    ### SANITY CHECK ###
    ####################

    # SANITY CHECK: Extract halt_motion timestamps directly from df (robot state data) and compare to halt_motion timestamps from log file
    halt_motion = df["halt_motion"].squeeze("columns")
    robot_state_data_halt_motion_start_robot_timestamps_us = df.iloc[
        df.index[halt_motion & ~halt_motion.shift(fill_value=False)]
    ]["CPU_steady_clock_time_us"].values.flatten()

    # Assert that the halt_motion timestamps from robot state data match those from the log file (sanity check)
    assert np.array_equal(
        robot_state_data_halt_motion_start_robot_timestamps_us, log_halt_motion_start_robot_timestamps_us
    ), "Halt motion timestamps from robot state data do not match those from the log file."

    ####################################################################
    ### Video Frames Corresponding to these Timepoints #################
    ####################################################################

    # Key timepoints (expressed in robot_timestamps_us)
    # trial_start_robot_timestamp_us
    # log_halt_motion_start_robot_timestamps_us (from log file)
    # log_wait_start_robot_timestamp_us (from log file) -- better to use this for the start of the grasp duration clock.
    # log_halt_motion_start_robot_timestamps_us (from log file) - when monkey grasps (can occur several times if monkey grasps/releases multiple times)
    # log_halt_motion_end_robot_timestamps_us (from log file) - when monkey releases

    frame_num_trial_start = 0
    frame_num_wait_start = int((log_wait_start_robot_timestamp_us - trial_start_robot_timestamp_us) / 1e6 * FPS)
    frame_num_wait_start_plus_1p5s = int(
        (log_wait_start_robot_timestamp_us + 1.5e6 - trial_start_robot_timestamp_us) / 1e6 * FPS
    )
    frame_num_final_halt_motion_end = int(
        (log_halt_motion_end_robot_timestamps_us[-1] - trial_start_robot_timestamp_us) / 1e6 * FPS
    )

    return frame_num_trial_start, frame_num_wait_start, frame_num_wait_start_plus_1p5s, frame_num_final_halt_motion_end

def process_session(session):
    # Get video trial name from anipose data
    anipose_dir = ANALYSIS_ROOT / session / "anipose" / "pose_3d_filter"
    anipose_fnames = glob(os.path.join(anipose_dir, "*.csv"))
    trial_names = get_trialnames(anipose_dir)

    # Get log and robot state dirs
    log_dir = DATA_ROOT / "Videos" / session / "trial_logs"
    robot_dir = DATA_ROOT / "Videos" / session / "robot_state_data"

    # Get corresponding log and robot state files
    log_fnames = find_log_or_robot(anipose_fnames, log_dir=log_dir, robot_dir=None)
    robot_fnames = find_log_or_robot(anipose_fnames, log_dir=None, robot_dir=robot_dir)

    data = []
    for trial_name, log_fname, robot_fname in zip(trial_names, log_fnames, robot_fnames):
        if log_fname == "nan" or robot_fname == "nan": continue
        robot_path = Path(robot_fname)
        trial_start, min_hold_start, min_hold_end, release = process_trial(log_fname, robot_path)
        new_data = [trial_name, trial_start, min_hold_start, min_hold_end, release, log_fname, robot_fname]
        data.append(new_data)

    df = pd.DataFrame(data, columns=[
        "trial_name", "trial_start_frame", "min_hold_start_frame", 
        "min_hold_end_frame", "release_frame", "log_file", "robot_file"
    ])

    output_dir = ANALYSIS_ROOT / session / "log"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir / "min_holding_window.csv", index=False)
    print(f"Saved {len(df)} trials to {output_dir / 'min_holding_window.csv'}")

def main():

    session_names = ["2025-08-19", "2025-08-22", "2025-11-19", "2025-11-20", "2025-12-04",
                        "2025-12-08", "2025-12-09", "2025-12-16", "2025-12-17", "2025-12-18"]

    for session_name in session_names:
        process_session(session_name)

if __name__ == "__main__":
    main() 
