##########################################################################################
# ShapedAcousticField_Simple.py — Object-Conformal Acoustic Field (Simplified Version)
##########################################################################################
#
# Uses the simplified radiation pressure model:
#   F = p²/(ρc²) · (-n̂) · A_face
#   v = p / (ρc)   (scalar speed, not a vector)
#
# This matches the force model in main.py.
# Compare with ShapedAcousticField.py (Gorkov version) for differences.
#
# Usage:
#   python ShapedAcousticField_Simple.py
#   python ShapedAcousticField_Simple.py --file ../3D_Files/cube_50mm.stl --scale 0.1
#
##########################################################################################
import numpy as np
import trimesh
import plotly.graph_objects as go
import os
import sys
import argparse
import time

# Ensure module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SimAcousticField import compute_complex_pressure, load_sources
from materials import get_medium, print_active_config, AMPLITUDE

##########################################################################################
# Configuration
##########################################################################################
DEFAULT_STL = os.path.join(os.path.dirname(__file__), '..', '3D_Files', 'Tetrahedron.stl')
DEFAULT_SCALE = 0.1
OFFSET_LAYERS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
GRID_RESOLUTION = 51
GRID_EXTENT = 25.0


##########################################################################################
# Mesh Loading (shared logic)
##########################################################################################
def load_mesh(filepath, scale=1.0):
    """Load a 3D mesh from .stl or .3mf file."""
    print(f"Loading mesh: {filepath}")
    mesh = trimesh.load(filepath)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"No valid geometry found in {filepath}")
    mesh.apply_scale(scale)
    mesh.vertices -= mesh.bounding_box.centroid
    print(f"  Faces: {len(mesh.faces)}, Vertices: {len(mesh.vertices)}, Volume: {mesh.volume:.4f} mm³")
    return mesh


##########################################################################################
# Surface Point Generation
##########################################################################################
def generate_surface_points(mesh, offsets=None):
    """Generate sample points on and near the object surface."""
    if offsets is None:
        offsets = OFFSET_LAYERS

    centroids = mesh.triangles_center
    normals = mesh.face_normals
    face_areas = mesh.area_faces
    M = len(centroids)

    all_points, all_normals_out, all_offsets = [], [], []
    for offset in offsets:
        all_points.append(centroids + offset * normals)
        all_normals_out.append(normals)
        all_offsets.append(np.full(M, offset))

    all_points = np.vstack(all_points)
    all_normals_out = np.vstack(all_normals_out)
    all_offsets = np.concatenate(all_offsets)

    print(f"  Surface sample points: {len(all_points)} "
          f"({M} centroids × {len(offsets)} layers)")
    return all_points, all_normals_out, all_offsets, centroids, face_areas


##########################################################################################
# Simplified Field Computation
##########################################################################################
def compute_simplified_field(points, normals, face_areas, sources, mesh, phases=None):
    """
    Compute the acoustic field using the simplified radiation pressure model.
    
    Pressure:   P = |Σ (A/r) exp(ikr)|
    Velocity:   v = P / (ρc)    (scalar)
    Rad. Press: p_rad = P² / (ρc²)
    Force:      F = p_rad · (-n̂) · A_face   (per face, on-surface only)
    """
    med = get_medium()
    rho = med["rho"]
    c = med["c"]

    N = len(points)
    N_faces = len(mesh.faces)

    print(f"\n  Computing pressure at {N} points...")
    t0 = time.time()
    p_complex = compute_complex_pressure(points, sources, phases)
    p_amplitude = np.abs(p_complex)
    print(f"    Done in {time.time() - t0:.2f}s")

    # Scalar velocity
    v_scalar = p_amplitude / (rho * c)

    # Radiation pressure
    p_rad = p_amplitude ** 2 / (rho * c ** 2)

    # Force per face (on-surface layer only, offset=0)
    offset_idx = OFFSET_LAYERS.index(0.0)
    start = offset_idx * N_faces
    end = start + N_faces

    p_rad_surface = p_rad[start:end]
    normals_surface = normals[start:end]
    areas_surface = face_areas  # face_areas is already just the on-surface count

    # F = p_rad * (-n̂) * A  — force pushes inward (against normal)
    force = p_rad_surface[:, np.newaxis] * (-normals_surface) * areas_surface[:, np.newaxis]

    return {
        'p_complex': p_complex,
        'p_amplitude': p_amplitude,
        'v_scalar': v_scalar,
        'p_rad': p_rad,
        'force': force,                    # (N_faces, 3) on-surface only
        'force_surface_start': start,
        'force_surface_end': end,
    }


##########################################################################################
# Volumetric Field
##########################################################################################
def compute_volumetric_field(sources, centroid, phases=None,
                              extent=GRID_EXTENT, resolution=GRID_RESOLUTION):
    """Compute pressure + velocity on a 3D grid centered on the object."""
    med = get_medium()
    rho = med["rho"]
    c = med["c"]

    x = np.linspace(centroid[0] - extent, centroid[0] + extent, resolution)
    y = np.linspace(centroid[1] - extent, centroid[1] + extent, resolution)
    z = np.linspace(centroid[2] - extent, centroid[2] + extent, resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    print(f"\n  Computing volumetric field on {resolution}³ grid...")
    t0 = time.time()
    p_complex = compute_complex_pressure(grid_points, sources, phases)
    print(f"    Volumetric pressure done in {time.time() - t0:.2f}s")

    p_vol = np.abs(p_complex).reshape(resolution, resolution, resolution)
    v_vol = (p_vol / (rho * c))
    p_rad_vol = p_vol ** 2 / (rho * c ** 2)

    return x, y, z, p_vol, v_vol, p_rad_vol


##########################################################################################
# Visualization
##########################################################################################
def plot_3d_surface_field(mesh, centroids, field_data, sources):
    """3D plot: mesh colored by pressure, simplified force arrows."""
    N_faces = len(centroids)
    s = field_data['force_surface_start']
    e = field_data['force_surface_end']

    p_surface = field_data['p_amplitude'][s:e]
    force = field_data['force']

    fig = go.Figure()

    # Mesh colored by pressure
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        intensity=p_surface, intensitymode='cell',
        colorscale='Viridis', colorbar=dict(title='Pressure (Pa)', x=1.0),
        opacity=0.7, flatshading=True, name='Pressure on Surface',
    ))

    # Force arrows (subsampled)
    step = max(1, N_faces // 200)
    idx = np.arange(0, N_faces, step)

    f_mags = np.linalg.norm(force, axis=1)
    f_avg = np.mean(f_mags[f_mags > 0]) if np.any(f_mags > 0) else 1.0
    f_scale = 3.0
    f_norm = force[idx] / f_avg * f_scale

    fx_lines, fy_lines, fz_lines = [], [], []
    for i, ci in enumerate(idx):
        fx_lines.extend([centroids[ci, 0], centroids[ci, 0] + f_norm[i, 0], None])
        fy_lines.extend([centroids[ci, 1], centroids[ci, 1] + f_norm[i, 1], None])
        fz_lines.extend([centroids[ci, 2], centroids[ci, 2] + f_norm[i, 2], None])

    fig.add_trace(go.Scatter3d(
        x=fx_lines, y=fy_lines, z=fz_lines,
        mode='lines', line=dict(color='#ff5555', width=3),
        name='Radiation Pressure Force',
    ))

    # Transducers
    fig.add_trace(go.Scatter3d(
        x=sources[:, 0], y=sources[:, 1], z=sources[:, 2],
        mode='markers', marker=dict(size=3, color='white', opacity=0.6),
        name='Transducers',
    ))

    limit = 50
    fig.update_layout(
        template='plotly_dark',
        title='Acoustic Field Mapped to Object Surface (Simplified Radiation Pressure)',
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


def plot_cross_sections(x, y, z, p_vol, v_vol, p_rad_vol, centroid):
    """Plot XY, XZ, YZ cross-section slices."""
    from matplotlib import pyplot as plt

    mid_x, mid_y, mid_z = len(x) // 2, len(y) // 2, len(z) // 2

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Acoustic Field Cross-Sections (Simplified)', fontsize=16, fontweight='bold')

    fields = [p_vol, v_vol, p_rad_vol]
    names = ['Pressure (Pa)', 'Velocity p/(ρc) (mm/s)', 'Radiation Pressure (Pa)']
    cmaps = ['viridis', 'plasma', 'inferno']

    for row, (vol, fname, cmap) in enumerate(zip(fields, names, cmaps)):
        ax = axes[row, 0]
        im = ax.imshow(vol[:, :, mid_z].T, extent=[x[0], x[-1], y[0], y[-1]],
                       origin='lower', cmap=cmap, aspect='equal')
        ax.set_title(f'{fname} — XY (z={z[mid_z]:.1f}mm)')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax)

        ax = axes[row, 1]
        im = ax.imshow(vol[:, mid_y, :].T, extent=[x[0], x[-1], z[0], z[-1]],
                       origin='lower', cmap=cmap, aspect='equal')
        ax.set_title(f'{fname} — XZ (y={y[mid_y]:.1f}mm)')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
        plt.colorbar(im, ax=ax)

        ax = axes[row, 2]
        im = ax.imshow(vol[mid_x, :, :].T, extent=[y[0], y[-1], z[0], z[-1]],
                       origin='lower', cmap=cmap, aspect='equal')
        ax.set_title(f'{fname} — YZ (x={x[mid_x]:.1f}mm)')
        ax.set_xlabel('Y (mm)'); ax.set_ylabel('Z (mm)')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


##########################################################################################
# Data Export
##########################################################################################
def export_csv(filepath, points, field_data, offsets):
    """Export computed field data to CSV."""
    data = np.column_stack([
        points,
        field_data['p_amplitude'],
        field_data['v_scalar'],
        field_data['p_rad'],
        offsets,
    ])
    header = 'x_mm,y_mm,z_mm,pressure_Pa,velocity_scalar_mm_s,radiation_pressure_Pa,offset_mm'
    np.savetxt(filepath, data, delimiter=',', header=header, comments='', fmt='%.6e')
    print(f"\n  CSV exported: {filepath}  ({len(points)} points)")


##########################################################################################
# Summary
##########################################################################################
def print_summary(field_data, mesh):
    """Print summary of simplified field."""
    s = field_data['force_surface_start']
    e = field_data['force_surface_end']
    p = field_data['p_amplitude'][s:e]
    v = field_data['v_scalar'][s:e]
    force = field_data['force']
    F_net = np.sum(force, axis=0)
    F_mags = np.linalg.norm(force, axis=1)

    print("\n" + "=" * 60)
    print("  FIELD SUMMARY — SIMPLIFIED MODEL (on-surface)")
    print("=" * 60)
    print(f"  Pressure:      min={np.min(p):.1f}  max={np.max(p):.1f}  mean={np.mean(p):.1f} Pa")
    print(f"  Velocity:      min={np.min(v):.2e}  max={np.max(v):.2e}  mean={np.mean(v):.2e} mm/s")
    print(f"  Force |F|:     min={np.min(F_mags):.4e}  max={np.max(F_mags):.4e}")
    print(f"  Net Force:     [{F_net[0]:.4e}, {F_net[1]:.4e}, {F_net[2]:.4e}]")
    print(f"  |Net Force|:   {np.linalg.norm(F_net):.4e}")
    print(f"  Object volume: {mesh.volume:.4f} mm³")
    print("=" * 60)


##########################################################################################
# Main
##########################################################################################
def main():
    parser = argparse.ArgumentParser(description='Object-Conformal Acoustic Field (Simplified)')
    parser.add_argument('--file', type=str, default=DEFAULT_STL,
                        help='Path to .stl or .3mf file')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE,
                        help='Mesh scale factor')
    parser.add_argument('--no-volume', action='store_true',
                        help='Skip volumetric field computation')
    parser.add_argument('--export', type=str, default='shaped_field_simple.csv',
                        help='CSV export filename')
    args = parser.parse_args()

    print_active_config()

    mesh = load_mesh(args.file, args.scale)
    sources = load_sources()
    print(f"\n  Transducer sources: {len(sources)}")

    print("\nGenerating surface sample points...")
    all_points, all_normals, all_offsets, centroids, face_areas = \
        generate_surface_points(mesh)

    print("\n--- Computing Acoustic Field (Simplified) ---")
    t_total = time.time()
    field_data = compute_simplified_field(
        all_points, all_normals, face_areas, sources, mesh)
    print(f"\n  Total computation: {time.time() - t_total:.2f}s")

    print_summary(field_data, mesh)

    csv_path = os.path.join(os.path.dirname(__file__), args.export)
    export_csv(csv_path, all_points, field_data, all_offsets)

    print("\nGenerating 3D surface plot...")
    plot_3d_surface_field(mesh, centroids, field_data, sources)

    if not args.no_volume:
        centroid = mesh.bounding_box.centroid
        x, y, z, p_vol, v_vol, p_rad_vol = compute_volumetric_field(
            sources, centroid)
        plot_cross_sections(x, y, z, p_vol, v_vol, p_rad_vol, centroid)


if __name__ == "__main__":
    main()
