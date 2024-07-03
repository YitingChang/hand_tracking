# Workflow for Lightning Pose

## Installation
1. Lightning Pose can be installed as a Python package on Linux.
2. Alternatively, create a Lightning.ai account to access Lightning Studio, a cloud-based environment with Lightning Pose installed.

Reference: https://lightning-pose.readthedocs.io/en/latest/source/installation.html

## Dataset preparation
Use DLC or JARVIS for annotation
### DLC to LP
For fused frames:
1. Fuse frames from multiple camera views and create a video (FuseFrames.ipynb)
2. Create a DeepLabCut project (DLC_training_Yiting.ipynb)
3. Extract frames  (DLC_training_Yiting.ipynb)
4. Label frames (DLC GUI)
5. Convert DLC labeled data to LP labeled data format (LP conversion tool)\
python scripts/converters/dlc2lp.py --dlc_dir=/home/yiting/Documents/DLC_projects/DLC_fused_2cam-Yiting-2024-01-12 --lp_dir=/home/yiting/Documents/LP_projects/LP_fused_2cam-Yiting-2024-01-012

### JARVIS to LP
1. Create a Jarvis project
2. Camera calibration
3. Extract frames
4. Label frames
5. Convert Jarvis labeled data to LP labeled data format ()

## Train LP model
6. Check video format (Check_VideoFormat.ipynb)
7. Get context frames (Get_ContextFrames.ipynb)
8. Edit LP configuration file
9. Train LP model (LP hydra)\
python scripts/train_hydra.py --config-path=/home/yiting/Documents/LP_projects/LP_fused_2cam-Yiting-2024-01-12 --config-name=config_hand-2cam-fused.yaml
