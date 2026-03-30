import numpy as np
import trimesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import os

def visualize_mesh_and_points(file_path, n_samples=1000, save_csv=True):
    """
    Reads a 3D mesh file (STL, 3MF, etc.), samples points uniformly across its surfaces,
    visualizes both, and optionally saves the points to a CSV.
    """
    # 1. Load the mesh file
    print(f"Loading {file_path}...")
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene):
            print("Detected scene with multiple geometries, concatenating...")
            mesh = mesh.to_geometry()
    except Exception as e:
        print(f"Error loading mesh file: {e}")
        return

    # 2. Sample points from the surface
    print(f"Sampling {n_samples} points from surface...")
    sampled_points, _ = trimesh.sample.sample_surface(mesh, n_samples)

    # 3. Save to CSV
    if save_csv:
        output_dir = "Scaffold_Points"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_name = os.path.join(output_dir, f"{base_name}_points.csv")
        
        print(f"Saving {n_samples} coordinates to {csv_name}...")
        np.savetxt(csv_name, sampled_points, delimiter=",", header="x,y,z", comments="")

    # 4. Visualization
    fig = plt.figure(figsize=(14, 8))
    
    # Plot 1: Mesh
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Generate unique colors for each triangle face
    n_faces = len(mesh.faces)
    face_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, n_faces))
    
    # Draw the mesh with individual face colors
    poly_collection = mplot3d.art3d.Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.5, edgecolor='black', linewidths=0.2)
    poly_collection.set_facecolor(face_colors)
    ax1.add_collection3d(poly_collection)
    
    # Plot triangle centroids matching the triangle colors (inverted for contrast)
    centroids = mesh.triangles_center
    centroid_colors = np.copy(face_colors)
    centroid_colors[:, :3] = 1.0 - centroid_colors[:, :3]  # Invert RGB
    
    ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                c=centroid_colors, marker='o', s=20, alpha=1.0)
    
    # Add a custom legend entry for centroids
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Inverse Centroids',
                              markerfacecolor='gray', markersize=8)]
    ax1.legend(handles=legend_elements)
    
    min_bound, max_bound = mesh.bounds
    center = (min_bound + max_bound) / 2
    max_dist = np.max(max_bound - min_bound) / 2
    
    ax1.set_xlim(center[0] - max_dist, center[0] + max_dist)
    ax1.set_ylim(center[1] - max_dist, center[1] + max_dist)
    ax1.set_zlim(center[2] - max_dist, center[2] + max_dist)
    
    ax1.set_title(f"Original Mesh ({os.path.basename(file_path)})", fontweight='bold')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")

    # Plot 2: Point Cloud
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 1. Plot sampled surface points
    pts = np.atleast_2d(sampled_points)
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                c='deepskyblue', marker='.', s=10, alpha=0.6, label='Sampled Surface Points')
    
    # 2. Plot original mesh vertices
    verts = mesh.vertices
    ax2.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
                c='crimson', marker='o', s=30, alpha=1.0, label='Original Mesh Vertices')
    
    ax2.legend()
    
    ax2.set_xlim(center[0] - max_dist, center[0] + max_dist)
    ax2.set_ylim(center[1] - max_dist, center[1] + max_dist)
    ax2.set_zlim(center[2] - max_dist, center[2] + max_dist)
    
    ax2.set_title(f"Surface Sampled Mold ({n_samples} points)", fontweight='bold')
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (mm)")

    print("Opening visualization window...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define directories
    input_dir = "3D_Files"
    # file_name = "PolySphere_Dodecahedron.3mf"
    file_name = "cube_50mm.stl"
    file_path = os.path.join(input_dir, file_name)
    
    if os.path.exists(file_path):
        visualize_mesh_and_points(file_path, n_samples=5000)
    else:
        print(f"File not found: {file_path}")
        print(f"Please ensure your file is inside the '{input_dir}' folder.")
