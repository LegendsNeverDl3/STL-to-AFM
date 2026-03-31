##########################################################################################
# ShapedAcousticField.py — Object-Conformal Acoustic Field (Gorkov Version)
##########################################################################################
#
# Computes the full acoustic field mapped to a 3D object's surface:
#   - Complex pressure → amplitude
#   - Particle velocity vector → speed
#   - Gorkov potential
#   - Acoustic radiation force (F = -∇U)
#
# Supports .stl (default) and .3mf files via trimesh.
#
# Usage:
#   python ShapedAcousticField.py
#   python ShapedAcousticField.py --file ../3D_Files/cube_50mm.stl --scale 0.1
#   python ShapedAcousticField.py --file ../3D_Files/PolySphere_Dodecahedron.3mf
#
##########################################################################################
import numpy as np
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import argparse
import time

# Ensure module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SimAcousticField import (
    compute_complex_pressure,
    compute_velocity_vector,
    compute_gorkov_potential,
    compute_gorkov_force,
    load_sources,
)
from materials import (
    get_medium, get_material, get_contrast_factors,
    print_active_config, AMPLITUDE,
)

##########################################################################################
# Configuration
##########################################################################################
DEFAULT_STL = os.path.join(os.path.dirname(__file__), '..', '3D_Files', 'cube_50mm.stl')
DEFAULT_SCALE = 0.1        # Scale factor for the mesh
OFFSET_LAYERS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # mm from surface along normals

# Volumetric grid for cross-sections
GRID_RESOLUTION = 51       # Points per axis for volumetric field
GRID_EXTENT = 25.0         # ±mm around object centroid


##########################################################################################
# Mesh Loading
##########################################################################################
def load_mesh(filepath, scale=1.0):
    """Load a 3D mesh from .stl or .3mf file."""
    print(f"Loading mesh: {filepath}")
    mesh = trimesh.load(filepath)

    # Handle Scene objects (.3mf files may load as scenes)
    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries into a single mesh
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"No valid geometry found in {filepath}")

    mesh.apply_scale(scale)
    mesh.vertices -= mesh.bounding_box.centroid  # Center at origin

    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Volume: {mesh.volume:.4f} mm³")
    print(f"  Bounding box: {mesh.bounds[0]} to {mesh.bounds[1]}")
    return mesh


##########################################################################################
# Object-Conformal Sample Point Generation
##########################################################################################
def generate_surface_points(mesh, offsets=None):
    """
    Generate sample points on and near the object surface.
    
    Points are placed at triangle centroids, offset along face normals
    at specified distances.
    
    Args:
        mesh:    trimesh.Trimesh object
        offsets: list of offset distances [mm] along normals (0.0 = on surface)
        
    Returns:
        all_points:  (N, 3) array of sample points
        all_normals: (N, 3) array of face normals for each point
        all_offsets: (N,) array of offset distances
        centroids:   (M, 3) array of face centroids (on-surface points only)
        face_areas:  (M,) array of face areas
    """
    if offsets is None:
        offsets = OFFSET_LAYERS

    centroids = mesh.triangles_center       # (M, 3)
    normals = mesh.face_normals             # (M, 3)
    face_areas = mesh.area_faces            # (M,)
    M = len(centroids)

    all_points = []
    all_normals = []
    all_offsets = []

    for offset in offsets:
        pts = centroids + offset * normals
        all_points.append(pts)
        all_normals.append(normals)
        all_offsets.append(np.full(M, offset))

    all_points = np.vstack(all_points)
    all_normals = np.vstack(all_normals)
    all_offsets = np.concatenate(all_offsets)

    print(f"  Surface sample points: {len(all_points)} "
          f"({M} centroids × {len(offsets)} offset layers)")

    return all_points, all_normals, all_offsets, centroids, face_areas


##########################################################################################
# Field Computation
##########################################################################################
def compute_shaped_field(points, sources, mesh_volume, phases=None):
    """
    Compute the full acoustic field at sample points: pressure, velocity, Gorkov, force.
    
    Args:
        points:      (N, 3) sample positions
        sources:     (M, 3) transducer positions
        mesh_volume: object volume [mm³]
        phases:      (M,) transducer phases [rad]
        
    Returns:
        dict with keys:
            p_complex, p_amplitude, v_vectors, v_speed,
            gorkov_U, gorkov_force
    """
    f1, f2 = get_contrast_factors()
    N = len(points)

    print(f"\n  Computing complex pressure at {N} points...")
    t0 = time.time()
    p_complex = compute_complex_pressure(points, sources, phases)
    p_amplitude = np.abs(p_complex)
    print(f"    Pressure done in {time.time() - t0:.2f}s")

    print(f"  Computing velocity vectors...")
    t0 = time.time()
    v_vectors, v_speed = compute_velocity_vector(points, sources, phases)
    print(f"    Velocity done in {time.time() - t0:.2f}s")

    print(f"  Computing Gorkov potential...")
    t0 = time.time()
    gorkov_U = compute_gorkov_potential(p_complex, v_speed, mesh_volume, f1, f2)
    print(f"    Gorkov potential done in {time.time() - t0:.2f}s")

    print(f"  Computing Gorkov force (F = -∇U)...")
    t0 = time.time()
    gorkov_force = compute_gorkov_force(points, sources, mesh_volume, f1, f2, phases)
    print(f"    Gorkov force done in {time.time() - t0:.2f}s")

    return {
        'p_complex': p_complex,
        'p_amplitude': p_amplitude,
        'v_vectors': v_vectors,
        'v_speed': v_speed,
        'gorkov_U': gorkov_U,
        'gorkov_force': gorkov_force,
    }


##########################################################################################
# Volumetric Field for Cross-Sections
##########################################################################################
def compute_volumetric_field(sources, centroid, mesh_volume, phases=None,
                              extent=GRID_EXTENT, resolution=GRID_RESOLUTION):
    """Compute pressure + velocity on a 3D grid centered on the object."""
    f1, f2 = get_contrast_factors()

    x = np.linspace(centroid[0] - extent, centroid[0] + extent, resolution)
    y = np.linspace(centroid[1] - extent, centroid[1] + extent, resolution)
    z = np.linspace(centroid[2] - extent, centroid[2] + extent, resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    print(f"\n  Computing volumetric field on {resolution}³ grid...")
    t0 = time.time()
    p_complex = compute_complex_pressure(grid_points, sources, phases)
    print(f"    Volumetric pressure done in {time.time() - t0:.2f}s")

    t0 = time.time()
    _, v_speed = compute_velocity_vector(grid_points, sources, phases)
    print(f"    Volumetric velocity done in {time.time() - t0:.2f}s")

    t0 = time.time()
    gorkov_U = compute_gorkov_potential(p_complex, v_speed, mesh_volume, f1, f2)
    print(f"    Volumetric Gorkov done in {time.time() - t0:.2f}s")

    p_vol = np.abs(p_complex).reshape(resolution, resolution, resolution)
    v_vol = v_speed.reshape(resolution, resolution, resolution)
    U_vol = gorkov_U.reshape(resolution, resolution, resolution)

    return x, y, z, p_vol, v_vol, U_vol


##########################################################################################
# Visualization
##########################################################################################
def plot_3d_surface_field(mesh, centroids, field_data, sources):
    """
    3D interactive plot: mesh colored by pressure, velocity arrows, force arrows.
    """
    # Get on-surface data (offset=0 layer)
    N_faces = len(centroids)
    # Find the on-surface layer (offset=0.0) — it's the middle layer
    offset_idx = OFFSET_LAYERS.index(0.0)
    start = offset_idx * N_faces
    end = start + N_faces

    p_surface = field_data['p_amplitude'][start:end]
    v_surface = np.abs(field_data['v_vectors'][start:end])
    v_speed_surface = field_data['v_speed'][start:end]
    force_surface = field_data['gorkov_force'][start:end]

    fig = go.Figure()

    # 1. Mesh colored by pressure
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        intensity=p_surface,
        intensitymode='cell',
        colorscale='Viridis',
        colorbar=dict(title='Pressure (Pa)', x=1.0),
        opacity=0.7,
        flatshading=True,
        name='Pressure on Surface',
    ))

    # 2. Velocity vectors as arrows on surface
    v_real = np.real(field_data['v_vectors'][start:end])
    v_mags = np.linalg.norm(v_real, axis=1)
    v_avg = np.mean(v_mags[v_mags > 0]) if np.any(v_mags > 0) else 1.0
    v_scale = 3.0  # visual scale

    # Subsample for clarity (show every nth arrow)
    step = max(1, N_faces // 200)
    idx = np.arange(0, N_faces, step)

    v_norm = v_real[idx] / v_avg * v_scale
    vx_lines, vy_lines, vz_lines = [], [], []
    for i, ci in enumerate(idx):
        vx_lines.extend([centroids[ci, 0], centroids[ci, 0] + v_norm[i, 0], None])
        vy_lines.extend([centroids[ci, 1], centroids[ci, 1] + v_norm[i, 1], None])
        vz_lines.extend([centroids[ci, 2], centroids[ci, 2] + v_norm[i, 2], None])

    fig.add_trace(go.Scatter3d(
        x=vx_lines, y=vy_lines, z=vz_lines,
        mode='lines', line=dict(color='#00e0ff', width=3),
        name='Velocity Vectors', showlegend=True,
    ))

    # 3. Gorkov force vectors
    f_mags = np.linalg.norm(force_surface, axis=1)
    f_avg = np.mean(f_mags[f_mags > 0]) if np.any(f_mags > 0) else 1.0
    f_scale = 3.0

    f_norm = force_surface[idx] / f_avg * f_scale
    fx_lines, fy_lines, fz_lines = [], [], []
    for i, ci in enumerate(idx):
        fx_lines.extend([centroids[ci, 0], centroids[ci, 0] + f_norm[i, 0], None])
        fy_lines.extend([centroids[ci, 1], centroids[ci, 1] + f_norm[i, 1], None])
        fz_lines.extend([centroids[ci, 2], centroids[ci, 2] + f_norm[i, 2], None])

    fig.add_trace(go.Scatter3d(
        x=fx_lines, y=fy_lines, z=fz_lines,
        mode='lines', line=dict(color='#ff5555', width=3),
        name='Gorkov Force', showlegend=True,
    ))

    # 4. Transducer positions
    fig.add_trace(go.Scatter3d(
        x=sources[:, 0], y=sources[:, 1], z=sources[:, 2],
        mode='markers', marker=dict(size=3, color='white', opacity=0.6),
        name='Transducers',
    ))

    limit = 50
    fig.update_layout(
        template='plotly_dark',
        title='Acoustic Field Mapped to Object Surface (Gorkov)',
        scene=dict(
            xaxis=dict(range=[-limit, limit], title='X (mm)'),
            yaxis=dict(range=[-limit, limit], title='Y (mm)'),
            zaxis=dict(range=[-limit, limit], title='Z (mm)'),
            aspectmode='cube',
        ),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05,
                    bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.show()


def plot_cross_sections(x, y, z, p_vol, v_vol, U_vol, centroid):
    """Plot XY, XZ, YZ cross-section slices of pressure, velocity, and Gorkov potential."""
    from matplotlib import pyplot as plt

    mid_x = len(x) // 2
    mid_y = len(y) // 2
    mid_z = len(z) // 2

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Acoustic Field Cross-Sections (Gorkov)', fontsize=16, fontweight='bold')

    # Row labels
    field_names = ['Pressure (Pa)', 'Velocity (mm/s)', 'Gorkov Potential']
    volumes = [p_vol, v_vol, U_vol]
    cmaps = ['viridis', 'plasma', 'coolwarm']

    for row, (vol, fname, cmap) in enumerate(zip(volumes, field_names, cmaps)):
        # XY slice (z = centroid)
        ax = axes[row, 0]
        im = ax.imshow(vol[:, :, mid_z].T, extent=[x[0], x[-1], y[0], y[-1]],
                       origin='lower', cmap=cmap, aspect='equal')
        ax.set_title(f'{fname} — XY (z={z[mid_z]:.1f}mm)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax)

        # XZ slice (y = centroid)
        ax = axes[row, 1]
        im = ax.imshow(vol[:, mid_y, :].T, extent=[x[0], x[-1], z[0], z[-1]],
                       origin='lower', cmap=cmap, aspect='equal')
        ax.set_title(f'{fname} — XZ (y={y[mid_y]:.1f}mm)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        plt.colorbar(im, ax=ax)

        # YZ slice (x = centroid)
        ax = axes[row, 2]
        im = ax.imshow(vol[mid_x, :, :].T, extent=[y[0], y[-1], z[0], z[-1]],
                       origin='lower', cmap=cmap, aspect='equal')
        ax.set_title(f'{fname} — YZ (x={x[mid_x]:.1f}mm)')
        ax.set_xlabel('Y (mm)')
        ax.set_ylabel('Z (mm)')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


##########################################################################################
# Data Export
##########################################################################################
def export_csv(filepath, points, field_data, offsets):
    """Export computed field data to CSV."""
    N = len(points)
    v_real = np.real(field_data['v_vectors'])

    data = np.column_stack([
        points,                              # x, y, z
        field_data['p_amplitude'],           # pressure
        v_real,                              # vx, vy, vz
        field_data['v_speed'],               # speed
        field_data['gorkov_U'],              # gorkov_U
        field_data['gorkov_force'],          # Fx, Fy, Fz
        offsets,                             # offset layer
    ])

    header = 'x_mm,y_mm,z_mm,pressure_Pa,vx_mm_s,vy_mm_s,vz_mm_s,speed_mm_s,gorkov_U,Fx,Fy,Fz,offset_mm'
    np.savetxt(filepath, data, delimiter=',', header=header, comments='', fmt='%.6e')
    print(f"\n  CSV exported: {filepath}  ({N} points)")


##########################################################################################
# Summary Statistics
##########################################################################################
def print_summary(field_data, mesh, offsets):
    """Print summary statistics of the computed field."""
    # On-surface data only (offset=0)
    N_faces = len(offsets) // len(OFFSET_LAYERS)
    idx_0 = OFFSET_LAYERS.index(0.0)
    s = idx_0 * N_faces
    e = s + N_faces

    p = field_data['p_amplitude'][s:e]
    v = field_data['v_speed'][s:e]
    U = field_data['gorkov_U'][s:e]
    F = field_data['gorkov_force'][s:e]
    F_net = np.sum(F, axis=0)
    F_mags = np.linalg.norm(F, axis=1)

    print("\n" + "=" * 60)
    print("  FIELD SUMMARY (on-surface values)")
    print("=" * 60)
    print(f"  Pressure:      min={np.min(p):.1f}  max={np.max(p):.1f}  mean={np.mean(p):.1f} Pa")
    print(f"  Velocity:      min={np.min(v):.2e}  max={np.max(v):.2e}  mean={np.mean(v):.2e} mm/s")
    print(f"  Gorkov U:      min={np.min(U):.4e}  max={np.max(U):.4e}")
    print(f"  Force |F|:     min={np.min(F_mags):.4e}  max={np.max(F_mags):.4e}")
    print(f"  Net Force:     [{F_net[0]:.4e}, {F_net[1]:.4e}, {F_net[2]:.4e}]")
    print(f"  |Net Force|:   {np.linalg.norm(F_net):.4e}")
    print(f"  Object volume: {mesh.volume:.4f} mm³")
    print("=" * 60)


##########################################################################################
# Main
##########################################################################################
def main():
    parser = argparse.ArgumentParser(description='Object-Conformal Acoustic Field (Gorkov)')
    parser.add_argument('--file', type=str, default=DEFAULT_STL,
                        help='Path to .stl or .3mf file')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE,
                        help='Mesh scale factor (default 0.1)')
    parser.add_argument('--no-volume', action='store_true',
                        help='Skip volumetric field computation (faster)')
    parser.add_argument('--export', type=str, default='shaped_field_gorkov.csv',
                        help='CSV export filename')
    args = parser.parse_args()

    # --- Config ---
    print_active_config()

    # --- Load mesh ---
    mesh = load_mesh(args.file, args.scale)

    # --- Load sources ---
    sources = load_sources()
    print(f"\n  Transducer sources: {len(sources)}")

    # --- Generate surface sample points ---
    print("\nGenerating surface sample points...")
    all_points, all_normals, all_offsets, centroids, face_areas = \
        generate_surface_points(mesh, OFFSET_LAYERS)

    # --- Compute field on surface ---
    print("\n--- Computing Acoustic Field (Gorkov) ---")
    t_total = time.time()
    field_data = compute_shaped_field(all_points, sources, mesh.volume)
    print(f"\n  Total surface field computation: {time.time() - t_total:.2f}s")

    # --- Summary ---
    print_summary(field_data, mesh, all_offsets)

    # --- Export CSV ---
    csv_path = os.path.join(os.path.dirname(__file__), args.export)
    export_csv(csv_path, all_points, field_data, all_offsets)

    # --- 3D Surface Plot ---
    print("\nGenerating 3D surface plot...")
    plot_3d_surface_field(mesh, centroids, field_data, sources)

    # --- Volumetric cross-sections ---
    if not args.no_volume:
        centroid = mesh.bounding_box.centroid
        x, y, z, p_vol, v_vol, U_vol = compute_volumetric_field(
            sources, centroid, mesh.volume)
        plot_cross_sections(x, y, z, p_vol, v_vol, U_vol, centroid)


if __name__ == "__main__":
    main()
