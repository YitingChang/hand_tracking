import os
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

