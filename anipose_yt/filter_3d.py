#!/usr/bin/env python3

from tqdm import tqdm, trange
import os.path, os
import numpy as np
import pandas as pd
from numpy import array as arr
from glob import glob
from scipy import signal
from scipy.interpolate import splev, splrep

from .common import make_process_fun, natural_keys

def medfilt_data(values, size=15):
    padsize = size+5
    vpad = np.pad(values, (padsize, padsize),
                  mode='median',
                  stat_length=5)
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_data(vals):
    nans, ix = nan_helper(vals)
    out = np.copy(vals)
    if np.mean(nans) > 0.85:
        return out
    out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    return out

def filter_pose(config, fname, outname):
    data = pd.read_csv(fname)

    cols = [x for x in data.columns if '_error' in x]
    bodyparts = [c.replace('_error', '') for c in cols]

    for bp in bodyparts:
        error = np.array(data[bp + '_error'])
        error[np.isnan(error)] = 100000
        bad = error > config['filter3d']['offset_threshold']
        for v in 'xyz':
            key = '{}_{}'.format(bp, v)
            values = np.array(data[key])
            values[bad] = np.nan
            values_intp = interpolate_data(values)
            values_filt = medfilt_data(values_intp,
                                       size=config['filter3d']['medfilt'])
            data[key] = values_filt
        data[bp+'_error'] = 10 # FIXME: hack for plotting
        
    data.to_csv(outname, index=False)


def process_session(config, session_path):
    pipeline_pose = config['pipeline']['pose_3d']
    pipeline_pose_filter = config['pipeline']['pose_3d_filter']

    pose_folder = os.path.join(session_path, pipeline_pose)
    output_folder = os.path.join(session_path, pipeline_pose_filter)

    pose_files = glob(os.path.join(pose_folder, '*.csv'))
    pose_files = sorted(pose_files, key=natural_keys)

    if len(pose_files) > 0:
        os.makedirs(output_folder, exist_ok=True)

    for fname in pose_files:
        basename = os.path.basename(fname)
        outpath = os.path.join(session_path,
                               pipeline_pose_filter,
                               basename)

        if os.path.exists(outpath):
            continue

        print(outpath)
        filter_pose(config, fname, outpath)

### Modify process_session to process our trial structure data ### 
# edited by Yiting on 2024/12/03
def process_trial(config, session_name, trial_name):
    pipeline_analysis_dir = config['pipeline']['analysis_dir']
    pipeline_pose = config['pipeline']['pose_3d']
    pipeline_pose_filter = config['pipeline']['pose_3d_filter']

    pose_folder = os.path.join(pipeline_analysis_dir, session_name,
                                "anipose", pipeline_pose)
    output_folder = os.path.join(pipeline_analysis_dir, session_name,
                                "anipose", pipeline_pose_filter)

    pose_file = os.path.join(pose_folder, trial_name + '_3d.csv')

    if len(pose_file) > 0:
        os.makedirs(output_folder, exist_ok=True)

    outpath = os.path.join(output_folder, trial_name + '_f3d.csv')

    if os.path.exists(outpath):
        return

    print(outpath)
    filter_pose(config, pose_file, outpath)

filter_pose_3d_all = make_process_fun(process_session)
