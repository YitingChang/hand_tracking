# Use Anipose to triangulate Lightning Pose 2D outputs 
## Anipose
Anipose is a toolkit for 3D pose estimation.It includes camera calibrations, triangulation, 2D/3D filters, and visualization. We can directly run Anipose or setup our own pipeline using a separate library, Aniposelib.  

**Anipose GitHub**: https://anipose.readthedocs.io/en/latest/ \
**Anipose paper**: https://doi.org/10.1016/j.celrep.2021.109730

## Lightning Pose to Anipose
We convert the 2d outputs of Lightning pose (.csv) to the inputs of Anipose (.h5):
1. Convert object to float
2. Create multi-level columns
3. Save in hdf fromat

## Anipose pipeline
1. Edit configuration files (.toml)
2. Camera calibration (intrinsic and extrinsic calibration)
2. 2D filters to refine or remove wrong detection  (optional)
3. Triangulation
4. 3D spatiotemporal filters to refine 3D pose (optional)
5. Visualization 
6. Analysis (position, velocity, joint angle etc.)

## Data structure
### Inputs
...experiment/config.toml
...experiment/session1/trial1/calibration_videos/cal_camA.mp4 \
...experiment/session1/trial1/calibration_videos/cal_camB.mp4 \
...experiment/session1/trial1/videos_raw/vid_camA.mp4 \
...experiment/session1/trial1/videos_raw/vid_camB.mp4
...experiment/session1/trial1/LP_pose_2d/vid_camA.h5
...experiment/session1/trial1/LP_pose_2d/vid_camB.h5


### Outputs
...experiment/session1/trial1/calibration_results/calibration.toml
...experiment/session1/trial1/pose_3d/vid.csv
...experiment/session1/trial1/videos_labeled_3d/vid.mp4
...experiment/session1/trial1/videos_combined/vid.mp4



