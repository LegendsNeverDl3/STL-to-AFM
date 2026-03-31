##########################################################################################
# SimAcousticField.py — Acoustic Field Computation Module
##########################################################################################
# Refactored into reusable functions. Can be imported by other scripts or run standalone.
#
# Functions:
#   compute_complex_pressure()   — vectorized coherent pressure at arbitrary points
#   compute_velocity_vector()    — particle velocity vector via pressure gradient
#   compute_gorkov_potential()   — Gorkov potential from pressure & velocity
#   compute_gorkov_force()       — acoustic radiation force F = -∇U
#
# Standalone: runs the original 81³ grid simulation with plotting.
##########################################################################################
import numpy as np
import time
import os
import sys

# Add parent directory so we can import materials when run standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from materials import get_medium, get_material, get_contrast_factors, AMPLITUDE, FREQUENCY

##########################################################################################
# Helper functions
##########################################################################################
def arrayprint(arr):
    for i in range(len(arr)):
        print(f"Source {i+1}: {arr[i]}")
    print("\n")


def load_sources(filepath=None):
    """Load transducer source positions from text file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srcarray.txt')
    rawsrc = np.loadtxt(filepath)
    return rawsrc.reshape(-1, 3)


##########################################################################################
# Core Acoustic Field Functions (Vectorized)
##########################################################################################

def compute_complex_pressure(points, sources, phases=None):
    """
    Compute the complex acoustic pressure field at arbitrary points.
    
    P(r) = Σᵢ (A/rᵢ) · exp(i·(k·rᵢ + φᵢ))
    
    Args:
        points:  (N, 3) array of field evaluation points [mm]
        sources: (M, 3) array of source positions [mm]
        phases:  (M,) array of source phases [rad], default all zeros
        
    Returns:
        (N,) complex array — complex pressure at each point
    """
    med = get_medium()
    k = med["k"]
    A = AMPLITUDE

    points = np.atleast_2d(points)
    sources = np.atleast_2d(sources)

    if phases is None:
        phases = np.zeros(len(sources))

    # Vectorized distance computation: (N, M)
    diff = points[:, np.newaxis, :] - sources[np.newaxis, :, :]  # (N, M, 3)
    r = np.linalg.norm(diff, axis=2)                              # (N, M)
    r = np.clip(r, 1e-3, None)

    # Complex pressure from each source: (N, M)
    angle = k * r + phases[np.newaxis, :]
    complex_fields = (A / r) * np.exp(1j * angle)

    # Coherent sum over all sources: (N,)
    total_pressure = np.sum(complex_fields, axis=1)
    return total_pressure


def compute_velocity_vector(points, sources, phases=None, delta=0.05):
    """
    Compute the acoustic particle velocity vector at each point.
    
    v⃗ = -∇P / (iωρ)
    
    Uses central finite differences to compute the pressure gradient.
    
    Args:
        points:  (N, 3) array of field evaluation points [mm]
        sources: (M, 3) array of source positions [mm]
        phases:  (M,) array of source phases [rad]
        delta:   finite difference step size [mm] (default 0.05 mm)
        
    Returns:
        velocity_vectors: (N, 3) complex velocity vector at each point [mm/s]
        speed:            (N,) float speed magnitude |v⃗| at each point [mm/s]
    """
    med = get_medium()
    omega = med["omega"]
    rho = med["rho"]

    points = np.atleast_2d(points)
    N = len(points)

    # Compute pressure gradient via central finite differences
    grad_p = np.zeros((N, 3), dtype=complex)
    for axis in range(3):
        # Forward and backward offset points
        pts_fwd = points.copy()
        pts_bwd = points.copy()
        pts_fwd[:, axis] += delta
        pts_bwd[:, axis] -= delta

        p_fwd = compute_complex_pressure(pts_fwd, sources, phases)
        p_bwd = compute_complex_pressure(pts_bwd, sources, phases)

        grad_p[:, axis] = (p_fwd - p_bwd) / (2.0 * delta)

    # Velocity: v = -∇P / (iωρ)
    velocity_vectors = -grad_p / (1j * omega * rho)

    # Speed: magnitude of velocity vector (using RMS for complex harmonic field)
    speed = np.sqrt(np.sum(np.abs(velocity_vectors) ** 2, axis=1))

    return velocity_vectors, speed


def compute_gorkov_potential(p_complex, v_speed, V0, f1=None, f2=None):
    """
    Compute the Gorkov potential at each point.
    
    U = V₀ · [ f₁ · ⟨p²⟩/(2ρ₀c₀²) − f₂ · (3ρ₀/4) · ⟨v²⟩ ]
    
    For harmonic fields:
        ⟨p²⟩ = |P|²/2   (time-averaged pressure squared)
        ⟨v²⟩ = |v|²/2   (time-averaged velocity squared)
    
    Args:
        p_complex: (N,) complex pressure field
        v_speed:   (N,) velocity magnitude (speed)
        V0:        object volume [mm³]
        f1, f2:    Gorkov contrast factors (default: from active material preset)
        
    Returns:
        U: (N,) float Gorkov potential at each point
    """
    if f1 is None or f2 is None:
        _f1, _f2 = get_contrast_factors()
        if f1 is None:
            f1 = _f1
        if f2 is None:
            f2 = _f2

    med = get_medium()
    rho_0 = med["rho"]
    c_0 = med["c"]

    # Time-averaged quantities for harmonic fields
    p_sq_avg = np.abs(p_complex) ** 2 / 2.0     # ⟨p²⟩
    v_sq_avg = v_speed ** 2 / 2.0                 # ⟨v²⟩

    # Gorkov potential
    U = V0 * (f1 * p_sq_avg / (2.0 * rho_0 * c_0 ** 2) - f2 * (3.0 * rho_0 / 4.0) * v_sq_avg)

    return U


def compute_gorkov_force(points, sources, V0, f1=None, f2=None, phases=None, delta=0.05):
    """
    Compute the acoustic radiation force from the Gorkov potential.
    
    F⃗ = -∇U
    
    Uses finite differences on U evaluated at offset points.
    
    Args:
        points:  (N, 3) field evaluation points [mm]
        sources: (M, 3) source positions [mm]
        V0:      object volume [mm³]
        f1, f2:  Gorkov contrast factors
        phases:  (M,) source phases [rad]
        delta:   finite difference step [mm]
        
    Returns:
        force: (N, 3) float force vector at each point [force units]
    """
    if f1 is None or f2 is None:
        _f1, _f2 = get_contrast_factors()
        if f1 is None:
            f1 = _f1
        if f2 is None:
            f2 = _f2

    points = np.atleast_2d(points)
    N = len(points)

    force = np.zeros((N, 3))

    for axis in range(3):
        # Forward offset
        pts_fwd = points.copy()
        pts_fwd[:, axis] += delta
        p_fwd = compute_complex_pressure(pts_fwd, sources, phases)
        _, v_fwd = compute_velocity_vector(pts_fwd, sources, phases, delta)
        U_fwd = compute_gorkov_potential(p_fwd, v_fwd, V0, f1, f2)

        # Backward offset
        pts_bwd = points.copy()
        pts_bwd[:, axis] -= delta
        p_bwd = compute_complex_pressure(pts_bwd, sources, phases)
        _, v_bwd = compute_velocity_vector(pts_bwd, sources, phases, delta)
        U_bwd = compute_gorkov_potential(p_bwd, v_bwd, V0, f1, f2)

        # F = -dU/dx via central difference
        force[:, axis] = -(U_fwd - U_bwd) / (2.0 * delta)

    return force


##########################################################################################
# Legacy standalone function (matches original behavior)
##########################################################################################
def acoustic_pressure_field_legacy(point_coords, sources):
    """
    Original single-point pressure function (kept for backward compatibility).
    """
    med = get_medium()
    k = med["k"]
    A = AMPLITUDE

    r = np.linalg.norm(sources - point_coords, axis=1)
    r = np.clip(r, 1e-3, None)

    angle = k * r
    complex_fields = (A / r) * (np.cos(angle) + 1j * np.sin(angle))
    total_field = np.sum(complex_fields)

    return np.abs(total_field)


##########################################################################################
# Standalone execution — original 81³ grid simulation
##########################################################################################
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    print("Loading source array...")
    srcarray = load_sources()
    arrayprint(srcarray)

    # Create simulation space
    grid_size = 81
    x = np.linspace(-40, 40, grid_size)
    y = np.linspace(-40, 40, grid_size)
    z = np.linspace(-40, 40, grid_size)

    # Vectorized computation on full grid
    print(f"Computing pressure on {grid_size}³ grid (vectorized)...")
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    start = time.time()
    p_complex = compute_complex_pressure(grid_points, srcarray)
    p_amplitude = np.abs(p_complex)
    simspace = p_amplitude.reshape(grid_size, grid_size, grid_size)
    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds.")

    # Plot XY slice
    plt.figure(figsize=(8, 6))
    mid = grid_size // 2
    plt.imshow(simspace[:, :, mid], extent=(-40, 40, -40, 40), origin='lower', cmap='viridis')
    plt.colorbar(label='Pressure Amplitude (Pa)')
    plt.title('Acoustic Pressure Field — XY Slice (z=0)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.xlim(-25, 25)
    plt.ylim(-30, 30)
    plt.clim(0, 4000)
    plt.show()
