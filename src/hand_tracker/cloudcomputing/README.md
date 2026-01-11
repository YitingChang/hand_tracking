# Rockfish Cluster
The Rockfish cluster is a computing resource available to researchers from Johns Hopkins University. We use the GPU nodes to train Lightning Pose models.   

Reference 
- [F&Q](https://www.arch.jhu.edu/support/faq/) 
- [User Guide](https://www.arch.jhu.edu/guide/)
## System Access
- Login to [coldfront](https://coldfront.rockfish.jhu.edu/) to manage account and allocations.
- Secure Shell (ssh) command: 

```
ssh -Y login.rockfish.jhu.edu -l userid
```

## Virtual Environment
- Create conda environment for Lightning Pose 
```
module load anaconda
conda create --name lgt-pose python=3.10
conda activate lgt-pose
pip install lightning-pose
```
- Activate conda environment 
```
module load anaconda
conda activate /path/to/env
```
environment path: /home/ychang73/.conda/envs/lgt-pose

Reference
- [Rockfish virtual environment](https://www.arch.jhu.edu/python-virtual-environments/)
- [Lightning Pose Installation](https://lightning-pose.readthedocs.io/en/latest/source/installation.html)

## Script and data management
- Download codes from [Lightning Pose Github](https://github.com/danbider/lightning-pose)
- Upload codes and data to VAST
- In the lightning-pose folder (~/lightning-pose-main), (1) put the project data (LP_240120) in the lightning-pose/data subfolder and (2) create a sbatch job script (hand_sbatch_1gpu.sh). 

~/lightning-pose-main/hand_sbatch_1gpu.sh \
~/lightning-pose-main/lightning-pose/data/LP_240120/labeled_data/... \
~/lightning-pose-main/lightning-pose/data/LP_240120/videos/... \
~/lightning-pose-main/lightning-pose/scripts/…


Reference
- [Rockfish Storage & Filesystems](https://www.arch.jhu.edu/support/storage-and-filesystems)
- [Lightning Pose Data Structure](https://lightning-pose.readthedocs.io/en/latest/source/user_guide/directory_structure.html)

## Submit job script
```
cd vast-doconn15/projects/hand_tracking/lightning-pose-main
sbatch hand_sbatch_1gpu.sh
```
## Commands
- Monitor submitted jobs
```
watch sqme
```
- Cancel all jobs
```
scancel -u user-email
```
- Cancel specific job
```
scancel job-id
```
 

