import os
from glob import glob
import pandas as pd
from datetime import datetime, timedelta
import json
import yaml

def load_json(json_path):
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_litpose_config(config_path):

    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_trialname(file_path):
    base = os.path.basename(file_path)
    filename, _ = os.path.splitext(base)
    filename_parts = filename.split('_')
    trialname = f"{filename_parts[0]}_{filename_parts[1]}"
    return trialname

def get_trialnames(folder_dir):
    trial_names = []
    filenames = [filename for filename in os.listdir(folder_dir)]
    for f in filenames:
        filename_parts = f.split('_')
        trial_name = filename_parts[0] + "_" + filename_parts[1]
        trial_names = trial_names + [trial_name]
    trial_names = sorted(trial_names)
    return trial_names

def load_log_trials(log_trials):
    logs = []
    for log_file in log_trials:
        log_data = load_json(log_file)
        log_data["trial_name"] = get_trialname(log_file)
        logs.append(log_data)
    logs = pd.DataFrame(logs)
    return logs

def get_video_timestamp(video_folder_path):
    base = os.path.basename(video_folder_path)
    base_parts = base.split('_')
    video_timestamp = base_parts[-1]
    return video_timestamp

def add_video_path(video_folder_paths, logs):
    video_folder_names = ["nan"] * logs.shape[0]
    trialnames = logs["trial_name"].tolist()

    for idx, trialname in enumerate(trialnames):

        # 1. Extract the timestamp part from the trial name
        # Parse it into a datetime object
        ts = datetime.strptime(trialname, "%Y-%m-%d_%H-%M-%S")

        # Shift by +1 second and -1 second
        ts_plus = ts + timedelta(seconds=1)
        ts_minus = ts - timedelta(seconds=1)

        # Convert back to the same string format
        ts_plus_str = ts_plus.strftime("%Y-%m-%d_%H-%M-%S")
        ts_minus_str = ts_minus.strftime("%Y-%m-%d_%H-%M-%S")

        # 2. Search for matching video folder names 
        matches = [v for v in video_folder_paths if trialname in v or ts_plus_str in v or ts_minus_str in v]
        if len(matches) > 1:
            print(f"Warning: Multiple matches found for trial {trialname}: {matches}")
        elif len(matches) == 1:
            video_folder_path = matches[0]
            # 3. Assign the found video folder name 
            video_folder_names[idx] = os.path.basename(video_folder_path)

    # 4. Add the video folder names to the logs DataFrame
    logs["video_folder_name"] = video_folder_names
    return logs

def find_matching_log(filenames, log_dir):
    """
    Finds the corresponding log file for a list of video/csv filenames,
    allowing for a +/- 1 second difference in timestamps.
    """
    matched_log_fnames = ["nan"] * len(filenames)
    
    # Get all log files from the directory
    log_fnames = glob(os.path.join(log_dir, "*.json"))

    for idx, fname in enumerate(filenames):
        # 0. Get trial name (timestamp string) from file name
        trialname = get_trialname(fname)

        try:
            # 1. Extract the timestamp part from the trial name
            # Parse it into a datetime object
            ts = datetime.strptime(trialname, "%Y-%m-%d_%H-%M-%S")

            # Shift by +1 second and -1 second
            ts_plus = ts + timedelta(seconds=1)
            ts_minus = ts - timedelta(seconds=1)

            # Convert back to the same string format
            ts_plus_str = ts_plus.strftime("%Y-%m-%d_%H-%M-%S")
            ts_minus_str = ts_minus.strftime("%Y-%m-%d_%H-%M-%S")

            # 2. Search for matching log file names
            # We look for the timestamp in the basename of the log file
            matches = [
                l for l in log_fnames 
                if trialname in os.path.basename(l) 
                or ts_plus_str in os.path.basename(l) 
                or ts_minus_str in os.path.basename(l)
            ]

            if len(matches) > 1:
                print(f"Warning: Multiple matches found for trial {trialname}: {matches}")
            elif len(matches) == 1:
                # 3. Assign the found log file path
                matched_log_fnames[idx] = matches[0]
                
        except ValueError:
            print(f"Skipping {fname}: Could not parse timestamp from '{trialname}'")
            continue

    return matched_log_fnames


