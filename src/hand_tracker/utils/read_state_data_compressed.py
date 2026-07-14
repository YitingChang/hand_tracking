from pathlib import Path
import json
from read_state_data import read_data_from_txt_file
import numpy as np

# Inputs
data_dir = Path("/media/yiting/NewVolume/Data/Videos")  # Folder containing dates of experiments
fname_log_json = "2025-12-09_09-01-20_log.json"  # log for specific trial in data_dir/YYYY_MM_DD/trial_logs dir

# Find corresponding robot_state_data file (within +/- 2s of the log file timestamp)
YYYY_MM_DD = fname_log_json.split("_")[0]
HH_MM_SS = fname_log_json.split("_")[1]

# Convert to seconds for comparison
HH, MM, SS = map(int, HH_MM_SS.split("-"))
HH_MM_SS_seconds = HH * 3600 + MM * 60 + SS
HH_MM_SS_seconds_minus_2s = HH_MM_SS_seconds - 2
HH_MM_SS_seconds_minus_1s = HH_MM_SS_seconds - 1
HH_MM_SS_seconds_plus_1s = HH_MM_SS_seconds + 1
HH_MM_SS_seconds_plus_2s = HH_MM_SS_seconds + 2

# Convert back to HH_MM_SS format
HH_MM_SS_minus_2s = f"{HH_MM_SS_seconds_minus_2s //
3600:02d}-{(HH_MM_SS_seconds_minus_2s % 3600) // 60:02d}-{HH_MM_SS_seconds_minus_2s % 60:02d}"
HH_MM_SS_minus_1s = f"{HH_MM_SS_seconds_minus_1s //
3600:02d}-{(HH_MM_SS_seconds_minus_1s % 3600) // 60:02d}-{HH_MM_SS_seconds_minus_1s % 60:02d}"
HH_MM_SS_plus_1s = f"{HH_MM_SS_seconds_plus_1s //
3600:02d}-{(HH_MM_SS_seconds_plus_1s % 3600) // 60:02d}-{HH_MM_SS_seconds_plus_1s % 60:02d}"
HH_MM_SS_plus_2s = f"{HH_MM_SS_seconds_plus_2s //
3600:02d}-{(HH_MM_SS_seconds_plus_2s % 3600) // 60:02d}-{HH_MM_SS_seconds_plus_2s % 60:02d}"

# Iterate through candidate times to find corresponding robot_state_data file
for candidate_time in [HH_MM_SS, HH_MM_SS_plus_1s, HH_MM_SS_minus_1s, HH_MM_SS_minus_2s, HH_MM_SS_plus_2s]:
    fname_robot_state_data_txt_gz = f"{YYYY_MM_DD}_{candidate_time}_state.txt.gz"
    compressed_file = Path(data_dir, YYYY_MM_DD, "robot_state_data", fname_robot_state_data_txt_gz)
    if compressed_file.exists():
        print(f"Found corresponding robot_state_data file: {compressed_file}")
        break
else:
    raise FileNotFoundError(
        "No corresponding robot_state_data file found within +/- 2 seconds of the log file timestamp."
    )

# Open compressed robot_state_data file and read its contents
df, timestamps, packet_loss_times, state_dict_list = read_data_from_txt_file(compressed_file)

# df is a dataframe with all robot data from that trial. It should correctly handle dropped packets
trial_start_robot_timestamp_us = df["CPU_steady_clock_time_us"].iloc[0].item()

# Read log file and extract halt_motion timestamps
with open(Path(data_dir, YYYY_MM_DD, "trial_logs", fname_log_json), "r") as f:
    log_data = json.load(f)
log_halt_motion_start_robot_timestamps_us = np.array(log_data["halt_motion_start_robot_timestamps_us"])

log_halt_motion_end_robot_timestamps_us = np.array(log_data["halt_motion_end_robot_timestamps_us"])


# Read log file and extract wait_start timestamps
log_wait_start_robot_timestamp_us = np.array(log_data["wait_start_robot_timestamp_us"])


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

FPS = 100.0  # hz

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

print("\n")
print("Key timepoints (expressed in video frame numbers):")
print(f"frame_num_trial_start: {frame_num_trial_start}")
print(f"frame_num_wait_start: {frame_num_wait_start}")
print(f"frame_num_wait_start_plus_1p5s: {frame_num_wait_start_plus_1p5s}")
print(f"frame_num_final_halt_motion_end: {frame_num_final_halt_motion_end}")
