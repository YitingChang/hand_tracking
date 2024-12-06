import os
import shutil
import cv2

### Trim video ###
def trim_video(input_file: str, output_file: str, duration):
    '''
    Args:
        input_file: the file path of video
        duration: the start and end times (in seconds) to trim the video
    Returns:
        output_file: the file path for saving the trimmed video     
    '''    

    vidcap = cv2.VideoCapture(input_file)
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    frame_start = int(duration[0]*frame_rate) 
    frame_end = int(duration[1]*frame_rate)

    video = cv2.VideoWriter(output_file,  #Provide a file to write the video to
                fourcc = cv2.VideoWriter_fourcc(*'mp4v'), # code for mp4
                fps=int(frame_rate),           #How many frames do you want to display per second in your video?
                frameSize=(width, height))                #The size of the frames you are writing

    for idx in range(frame_start, frame_end):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx) # Set the frame to get
        ref, frame = vidcap.read()
        video.write(frame)

    cv2.destroyAllWindows()

### Downsample video ###
def downsample_video(input_file: str, output_file: str, downsample_factor):
    '''
    Args:
        input_file: the file path of video
        downsample_factor: # of frames of downsampled video = # of frames of video / downsample_factor
    Returns:
        output_file: the file path for saving the downsampled video     
    '''    

    vidcap = cv2.VideoCapture(input_file)
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    total_frame_number = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    video = cv2.VideoWriter(output_file,  #Provide a file to write the video to
                fourcc = cv2.VideoWriter_fourcc(*'mp4v'), # code for mp4
                fps=int(frame_rate/downsample_factor),    #How many frames do you want to display per second in your video?
                frameSize=(width, height))                #The size of the frames you are writing

    for idx in range(0, total_frame_number, downsample_factor):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx) # Set the frame to get
        ref, frame = vidcap.read()
        video.write(frame)

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    ## Setting 
    folder_path = r'/home/yiting/Documents/Anipose_projects/Anipose_5cam_241021/Anipose_241021/2024-10-21_16-19-18/calibration_videos_org'
    trimmed_folder_path = r'/home/yiting/Documents/Anipose_projects/Anipose_5cam_241021/Anipose_241021/2024-10-21_16-19-18/calibration_videos'
    cameras = os.listdir(folder_path) 
    # Set the start and end times (in seconds) to trim the video
    duration = [11, 22]

    for c in cameras:
        input_path = os.path.join(folder_path, c)
        output_path = os.path.join(trimmed_folder_path, c)
        trim_video(input_path, output_path, duration)
