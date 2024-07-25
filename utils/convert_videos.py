"""Functions for preparing data to train lightning pose models"""  
import os
import subprocess
import shutil

def contains_subdirectory(directory):
    for root, directories, files in os.walk(directory):
        if directories:
            return True
    return False  

### Check video format and reencode video ###
def check_codec_format(input_file: str) -> bool:
    """Run FFprobe command to get video codec and pixel format."""
    ffmpeg_cmd = f'ffmpeg -i {input_file}'
    output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    # stderr because the ffmpeg command has no output file, but the stderr still has codec info.
    output_str = output_str.stderr
    # search for correct codec (h264) and pixel format (yuv420p)
    if output_str.find('h264') != -1 and output_str.find('yuv420p') != -1:
        # print('Video uses H.264 codec')
        is_correct_format = True
    else:
        is_correct_format = False
    return is_correct_format

def reencode_video(input_file: str, output_file: str) -> None:
    """reencodes video into H.264 coded format using ffmpeg from a subprocess.

    Args:
        input_file: abspath to existing video
        output_file: abspath to to new video

    """
    # check input file exists
    assert os.path.isfile(input_file), "input video does not exist."
    # check directory for saving outputs exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ffmpeg_cmd = f'ffmpeg -i {input_file} -c:v libx264 -pix_fmt yuv420p -c:a copy -y {output_file}'
    subprocess.run(ffmpeg_cmd, shell=True)

def get_videos(jarvis_dir, lp_dir, src_vid_dir):
    os.makedirs(os.path.join(lp_dir,'videos'), exist_ok=True)
    # Find trial folders in JARVIS labeled dataset 
    trials = [filename for filename in os.listdir(jarvis_dir) if contains_subdirectory(os.path.join(jarvis_dir,filename))]
    trials.sort()

    for t in trials:
        trialname_parts = t.split('_')
        cameras = os.listdir(os.path.join(src_vid_dir,trialname_parts[0], t))
        for c in cameras:
            vidname_new = trialname_parts[0] + 'T' + trialname_parts[1] + '_' + c
            # Check video format and reencode video (if needed)
            if check_codec_format(os.path.join(src_vid_dir, t, c)):
                shutil.copy(os.path.join(src_vid_dir, trialname_parts[0], t, c), os.path.join(lp_dir, 'videos', vidname_new))
            else:
                reencode_video(os.path.join(src_vid_dir, trialname_parts[0], t, c), os.path.join(lp_dir, 'videos', vidname_new))




