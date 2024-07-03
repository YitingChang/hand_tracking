# Use Anipose to triangulate Lightning Pose 2D outputs 
## Anipose
Anipose is a toolkit for 3D pose estimation.It includes camera calibrations, triangulation, 2D/3D filters, and visualization. We can directly run Anipose or setup our own pipeline using a separate library, Aniposelib.  

**Anipose GitHub**: https://anipose.readthedocs.io/en/latest/ \
**Anipose paper**: https://doi.org/10.1016/j.celrep.2021.109730

## Lightning Pose to Anipose
We convert the 2d outputs of Lightning pose to the inputs of Anipose as follow steps:
1. Convert object to float
2. Create multi-level columns
3. Save in hdf fromat

## Anipose pipeline
1. Camera calibration (intrinsic and extrinsic calibration)
2. 2D filters to refine or remove wrong detection  (optional)
3. Triangulation
4. 3D spatiotemporal filters to refine 3D pose (optional)
5. Visualization 
6. Analysis (position, velocity, joint angle etc.)