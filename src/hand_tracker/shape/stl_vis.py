import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

# 1. Load the mesh
# Note: Ensure the path is correct for your Linux mount
stl_path = "/media/yiting/NewVolume/Data/Shapes/shapes_stl/A013.stl"
test_mesh = mesh.Mesh.from_file(stl_path)

# 2. Create the figure and axis correctly
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3. Create the 3D collection
# alpha sets transparency; facecolors/edgecolors make the geometry visible
poly_collection = mplot3d.art3d.Poly3DCollection(test_mesh.vectors, 
                                                 alpha=0.8, 
                                                 facecolors='cyan', 
                                                 edgecolors='black', 
                                                 linewidths=0.1)

# 4. Add to plot
ax.add_collection3d(poly_collection)

# 5. Auto-scale the limits
# Matplotlib 3D doesn't auto-scale Collections, so we do it manually
scale = test_mesh.points.flatten()
ax.auto_scale_xyz(scale, scale, scale)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title(f"Visualizing {stl_path.split('/')[-1]}")
plt.show()