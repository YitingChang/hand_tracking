# hand_tracking
In Dr. Daniel O'Connor's lab, we are interested in 3D shape perception from touch in primates. We use our hands to grasp, recognize, and manipulate objects. To understand how we perceive 3D shapes using tactile signals, it is critical to track how hands interact with objects. This repository is created for markerless 3D tracking of monkey hand from multiple camera views. It leverages the machine learning approach **DeepLabCut** to track keypoints in 2D and **NCams** to estimate 3D pose. 

DeepLabCut is a 2D CNN. Currently, I'm exploring other 3D tracking tools that use 3D CNNs ([DANNCE](https://github.com/spoonsso/dannce)) or hybrid 2D/3D CNNs [JARVIS](https://github.com/JARVIS-MoCap/JARVIS-HybridNet). I will compare the performance of different 3D tracking tools to determine which networks work better in our study. 

## Prerequisites
- **Software**
  - DeepLabCut (https://deeplabcut.github.io/DeepLabCut/docs/UseOverviewGuide.html)
  - NCams (https://github.com/CMGreenspon/NCams)
  - Note:
    - [3D DeepLabCut](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html) only supports 2-camera based 3D pose estimation as of October 31st, 2023. Therefore, we use NCams for 3D tracking from more than 2 cameras.
    - NCams is a toolbox to use multiple cameras to track and reconstruct the kinematics of primate limbs. As of October 31st, 2023, this repository uses the modules of camera calibration and triangulation. The module for musculoskeletal modeling based on [OpenSIM](https://simtk.org/frs/index.php?group_id=91#package_id319) may be useful. Consider to add it to the processing pipeline later.  
    - Other 3D tracking tools: [Anipose](https://anipose.readthedocs.io/en/latest/), [Lightning Pose](https://github.com/danbider/lightning-pose), [DANNCE](https://github.com/spoonsso/dannce), and [JARVIS](https://github.com/JARVIS-MoCap/JARVIS-HybridNet)
- **Hardware**\
To study 3D shape perception from touch, experiments are designed to be conducted in the dark to minimize visual information of 3D objects. Therefore, in this study, infrared illuminators and cameras are used to capture images in the dark. 
  - Cameras:\
    FLIR Blackfly S3 cameras (4 [mono-color](https://www.edmundoptics.com/p/bfs-u3-23s3m-c-usb3-blackflyreg-s-monochrome-camera/41346/#) cameras with [25mm focal length lens](https://www.edmundoptics.com/p/25mm-uc-series-fixed-focal-length-lens/2971/) and 2 [color](https://www.edmundoptics.com/p/bfs-u3-23s3c-c-usb3-blackflyreg-s-color-camera/41347/) cameras with [12mm focal length lens](https://www.edmundoptics.com/p/12mm-uc-series-fixed-focal-length-lens/2969/)).\
    [Remove IR filters](https://www.flir.com/support-center/iis/machine-vision/knowledge-base/removing-the-ir-filter-from-a-color-camera/ ) if needed.
  - Illuminators:\
    [Edmundoptics IR Spot Lights](https://www.edmundoptics.com/f/advanced-illumination-long-working-distance-high-intensity-spot-lights/39791/) (940 nm)
  
## Calibration
- **Calibration Images**
  - Print a checker or charuco board.
  - For intrinsic calibration, take images for individial cameras. 
  - For extrinsic calibration, take images for all cameras.
  - Note:
    - A charuco board is recommended. See an example [6*8 charuco board](charuco_board_6x8.pdf).
    - We can use `ncams.camera_tools.create_board` to create a checker or charuco board (see Step 2 in [Camera_calibration_pipeline](Camera_calibration_pipeline.ipynb)).
    - The function for creating a checker board in NCams does not work for an even-number board dimension. This bug was fixed in [Create_checkerboard](Create_checkerboard.ipynb).
    - It is recommended to take about 50-70 sets of images. While taking images, try to cover several distances and all parts of views.
    - For checker board calibration, keep the orientation consistent. Do not roate the checker board more than 30 degrees.
    - More tips can be found [here](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html). 

- **Camera Calibration**
\
  [Camera_calibration_pipeline](Camera_calibration_pipeline.ipynb) is an example script for camera calibration.\
  It includes the following steps:
  - Creating a configuration file and setting up file structures
  - Creating a charuco or checker board
  - Camera intrinsic calibration
  - Camera extrinsic calibration
     - One-shot multi PnP (if all cameras can capture the board at the same time)
     - Sequential-stereoc (if not all cameras can capture the board at the same time)
  - Loading an existing setup
  - Calibrating an individual camera

## 3D Tracking
- **Video Recording**
  - Cameras are set up to cover all key points of monkey hand, and each key point is viewed by at least 2 cameras.
  - Show (1) experimental setup (camera location etc.) and (2) a set of images from all the cameras.
  - Synchronization of multiple cameras. (Ask William)
  - Frame size and frame rate.
  
- **Network Training**\
  We use **Deeplabcut** for network training. Specifically, a single network is trained to track the key points of monkey hands from all camera views. (See [DLC_traning_Yiting](DLC_traning_Yiting.ipynb))
  - We may want to improve the accuracy using different filters for temporal and spatial constraints. Or try pre-trained hand models. 
  - Points of interests (Insert an labeled image)
  
- **Triangulation and Plotting**
  - Import DLC results (.csv) and labeled videos.\
    \
    Data structure:
    - Project/Animal/Session Folder
      - Trial_N
          - Trial_N_cam12345678_DLC.csv
          - Trial_N_cam23456789_DLC.csv

  - [Triangulation and plotting](Triangulation_and_Plotting.ipynb) 

