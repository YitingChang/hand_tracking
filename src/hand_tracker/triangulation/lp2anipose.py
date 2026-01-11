### Convert Lightning Pose 2d outputs (.csv) to Anipose inputs (.hdf)
# 1. Convert object to float
# 2. Create multi-level columns

import os
import pandas as pd
import numpy

def get_trial_names(lp_dir):
    trial_names = []
    filenames = [filename for filename in os.listdir(lp_dir) if filename.endswith('_camTo.csv')]
    for f in filenames:
        filename_parts = f.split('_')
        trial_name = filename_parts[0] + "_" + filename_parts[1]
        trial_names = trial_names + [trial_name]
    return trial_names

def lp2anipose(lp_path, anipose_path):
    # df = pd.read_csv(lp_path, header = None, index_col = 0)
    # # Convert object data to float data
    # arr = df.iloc[3:].to_numpy()
    # new_arr = arr.astype('f')
    # new_df = pd.DataFrame(data=new_arr)
    # # Create multi-level index for columns
    # column_arr = df.iloc[0:3].to_numpy() 
    # tuples = list(zip(*column_arr))
    # new_df.columns = pd.MultiIndex.from_tuples(tuples, names=df.index[0:3])
    # # Save in hdf format
    # new_df.to_hdf(anipose_path, key = 'new_df', mode='w')  

    df = pd.read_csv(lp_path, header=[0, 1, 2], index_col=0)
    df.to_hdf(anipose_path, key='new_df', mode='w')

def lp2anipose_session(lp_dir, anipose_dir, camera_views):
    trials = get_trial_names(lp_dir)
    for t in trials:
        anipose_trial_dir = os.path.join(anipose_dir, t)
        os.makedirs(anipose_trial_dir, exist_ok = True)
        for c in camera_views:
            lp_file_path = os.path.join(lp_dir, t + "_cam" +  c + ".csv")
            anipose_file_path = os.path.join(anipose_trial_dir, t + "_cam" +  c + ".h5")
            lp2anipose(lp_file_path, anipose_file_path)


 