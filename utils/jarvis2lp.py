### Convert Jarvis labeled data to Lightning Pose labeled data ###
import glob
import os
import shutil
import pandas as pd
import cv2
import numpy as np
from PIL import Image

def contains_subdirectory(directory:str):
    for root, directories, files in os.walk(directory):
        if directories:
            return True
    return False   

def J2LP_csv(csv_file, t, c):
    '''
    Args:
        csv_file: Jarvis annotation file (.csv) for a single video
        t: trial video name
        c: camera view name
    Returns:
        df_tmp: Lightning Pose annotation     
    '''
    df1 = pd.read_csv(csv_file, on_bad_lines='skip', header = None, index_col=0) 
    df2 = pd.read_csv(csv_file, skiprows=4, header = None, index_col=0) 
    last_column = df2.shape[1]
    df2 = df2.drop(columns=[last_column],axis='columns')

    # Remove entities row
    isNotEntities = [x != 'entities' for x in df1.index.values]
    df1 = df1.iloc[isNotEntities]
    # Find coords row and remove state columns
    isCoords = [x == 'coords' for x in df1.index.values]
    isXY = [s != 'state' for s in df1.iloc[isCoords].values]
    df1 = df1.iloc[:,isXY[0]]
    df2 = df2.iloc[:, isXY[0]]

    # Replace image file name with its file path
    imgs = list(df2.index.values)
    new_col = []
    trialname_parts = t.split('_')
    trialname_new = trialname_parts[0] + 'T' + trialname_parts[1]
    for i in imgs:
        root, ext = os.path.splitext(i)
        framename_parts = root.split('_')
        framename_new = 'img' + format(int(framename_parts[1]), '04d') + '.png' # Change .jpg to .png (JARVIS- .jpg, LP/DLC- .png)
        framepath = f"labeled-data/{trialname_new}/{c}/{framename_new}"
        new_col.append(framepath)

    df2.index = new_col
    df_tmp = pd.concat([df1,df2])
    return df_tmp

def get_labeledframes(jarvis_dir:str, lp_dir:str):
    # Find trial folders in JARVIS labeled dataset 
    trials = [filename for filename in os.listdir(jarvis_dir) if contains_subdirectory(os.path.join(jarvis_dir,filename))]
    trials.sort()

    # Get camera view names
    cameras = os.listdir(os.path.join(jarvis_dir, trials[0]))

    # Copy labeled frames to the LP project folder
    for t in trials: # trial video name
        for c in cameras: # camera view name
            trialname_parts = t.split('_')
            trialname_new = trialname_parts[0] + 'T' + trialname_parts[1]
            os.makedirs(os.path.join(lp_dir,"labeled-data",trialname_new, c), exist_ok=True)
            # Convert .jpg to .png and copy frames over
            imgs = [im for im in os.listdir(os.path.join(jarvis_dir,t,c)) if im.endswith('.jpg')]
            for i in imgs:
                im = Image.open(os.path.join(jarvis_dir,t,c,i))
                root, ext = os.path.splitext(i)
                framename_parts = root.split('_')
                framename_new = 'img' + format(int(framename_parts[1]), '04d') + '.png' # Change .jpg to .png (JARVIS- .jpg, LP/DLC- .png)
                im.save(os.path.join(lp_dir,"labeled-data",trialname_new, c, framename_new))

    
def J2LP_sigview(jarvis_dir:str, lp_dir:str):
    # Find trial folders in JARVIS labeled dataset 
    trials = [filename for filename in os.listdir(jarvis_dir) if contains_subdirectory(os.path.join(jarvis_dir,filename))]
    trials.sort()
    dfs = []
    for t in trials:
        # Get camera view names
        cameras = os.listdir(os.path.join(jarvis_dir, t))
        for c in cameras:
            csv_file = glob.glob(os.path.join(jarvis_dir, t, c, "annotations.csv"))[0]
            df_tmp = J2LP_csv(csv_file, t, c)
            df_tmp.to_csv(os.path.join(jarvis_dir,t,c, "CollectedData.csv"), header = False)
            df = pd.read_csv(os.path.join(jarvis_dir,t,c, "CollectedData.csv"), header = [0,1,2], index_col=0)
            dfs.append(df)
    df_all = pd.concat(dfs)  

    os.makedirs(lp_dir, exist_ok=True)
    # Save concatenated labels
    df_all.to_csv(os.path.join(lp_dir, "CollectedData.csv"))


def J2LP_mulview(jarvis_dir:str, lp_dir:str):
    # Find trial folders in JARVIS labeled dataset 
    trials = [filename for filename in os.listdir(jarvis_dir) if contains_subdirectory(os.path.join(jarvis_dir,filename))]
    trials.sort()

    # Get camera view names
    cameras = os.listdir(os.path.join(jarvis_dir, trials[0]))

    # Convert Jarvis labeled data to Lightning Pose labeled data
    for c in cameras:
        dfs = []
        for t in trials: 
            csv_file = glob.glob(os.path.join(jarvis_dir, t, c, "annotations.csv"))[0]
            df_tmp = J2LP_csv(csv_file, t, c)
            df_tmp.to_csv(os.path.join(jarvis_dir,t,c, "CollectedData.csv"), header = False)
            df = pd.read_csv(os.path.join(jarvis_dir,t,c, "CollectedData.csv"), header = [0,1,2], index_col=0)
            dfs.append(df)
        df_all = pd.concat(dfs)
        os.makedirs(lp_dir, exist_ok=True)

        # Save concatenated labels for each camera view
        df_all.to_csv(os.path.join(lp_dir, c + ".csv"))

def J2LP_sigview_multisession(jarvis_annotations_list:list, lp_dir:str):
    dfs = []
    for jarvis_dir in jarvis_annotations_list:
        # Find trial folders in JARVIS labeled dataset 
        trials = [filename for filename in os.listdir(jarvis_dir) if contains_subdirectory(os.path.join(jarvis_dir,filename))]
        trials.sort()
        for t in trials:
            # Get camera view names
            cameras = os.listdir(os.path.join(jarvis_dir, t))
            for c in cameras:
                csv_file = glob.glob(os.path.join(jarvis_dir, t, c, "annotations.csv"))[0]
                df_tmp = J2LP_csv(csv_file, t, c)
                df_tmp.to_csv(os.path.join(jarvis_dir,t,c, "CollectedData.csv"), header = False)
                df = pd.read_csv(os.path.join(jarvis_dir,t,c, "CollectedData.csv"), header = [0,1,2], index_col=0)
                dfs.append(df)
    df_all = pd.concat(dfs)  

    os.makedirs(lp_dir, exist_ok=True)
    # Save concatenated labels
    df_all.to_csv(os.path.join(lp_dir, "CollectedData.csv"))

def get_contextframes(lp_dir, context_range):
    videos = os.listdir(os.path.join(lp_dir, "videos"))
    for v in videos:
        video = cv2.VideoCapture(os.path.join(lp_dir, "videos", v))
        root, ext = os.path.splitext(v)
        vidname_parts = root.split('_')
        trialname = vidname_parts[0] 
        viewname = vidname_parts[1]
        frame_folder = os.path.join(lp_dir, "labeled-data", trialname, viewname)
        frame_idxs = [
            int(f[3:len(f)-4])
            for f in os.listdir(frame_folder)
            if f.endswith('.png')
        ]
        # Get context frames
        for fr_idx in frame_idxs:
            fr_con = np.arange(fr_idx + context_range[0], fr_idx + context_range[1] + 1)
            for i in fr_con:
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                img_save_path = os.path.join(frame_folder, 'img' + format(i,'04d') + '.png')
                cv2.imwrite(img_save_path, frame)

    



