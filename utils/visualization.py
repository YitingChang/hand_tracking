# Visualization: video + feature traces across time 

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Generate a figure for the traces
def create_traces_frame(time, traces_data, feature_columns):
    plt.figure(figsize=(8, 4))
    for feature in feature_columns:
        plt.plot(traces_data['time'], traces_data[feature], label=feature)
    plt.axvline(x=time, color='r', linestyle='--', label='Current Time')  # Moving bar
    plt.xlabel('Time')
    plt.ylabel('Angle (degree)')
    plt.title('Angle Traces')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig('current_frame.png')
    plt.close()
    traces_frame = cv2.imread('current_frame.png')
    traces_frame = cv2.cvtColor(traces_frame, cv2.COLOR_BGR2RGB)
    return traces_frame

def creat_combined_video(video_path, traces_csv, output_path, feature_columns):

    # Load traces data
    traces_data = pd.read_csv(traces_csv)

    # Load video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the output video writer
    output_height = frame_height + 300  # Adjust for trace height
    output_width = frame_width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # Create a time column
    traces_data['time'] = traces_data['fnum'] / fps

    # Process each frame
    for frame_idx in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        # Calculate current time
        current_time = frame_idx / fps

        # Generate trace frame
        traces_frame = create_traces_frame(current_time, traces_data, feature_columns)

        # Combine video frame (upper) and trace frame (lower)
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        combined_frame[:frame_height, :, :] = frame
        combined_frame[frame_height:, :, :] = cv2.resize(traces_frame, (output_width, 300))

        # Write the frame
        out_video.write(combined_frame)

    # Release resources
    video.release()
    out_video.release()

    print(f"Output video saved to {output_path}")