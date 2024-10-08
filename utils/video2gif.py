# Convert videos to gif
from moviepy.editor import VideoFileClip

# Load the video file
video_clip = VideoFileClip("/home/yiting/Documents/Data/Videos/FusedVideos/2024-04-15_10-36-24_969748_resized5.mp4")

# Save as GIF (1-5 sec clip)
video_clip.subclip(1, 5).write_gif("/home/yiting/Documents/Data/Videos/FusedVideos/gif/2024-04-15_10-36-24_969748_resized5.gif", fps=10)

