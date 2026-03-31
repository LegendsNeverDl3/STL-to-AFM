##########################################################################################
# materials.py — Material Preset System for Acoustic Field Modeling
##########################################################################################
#
# Switch materials by changing ACTIVE_MATERIAL and ACTIVE_MEDIUM below.
# All downstream code reads from get_medium(), get_material(), get_contrast_factors().
#
##########################################################################################
import numpy as np

##########################################################################################
# Medium Properties (the fluid the sound travels through)
##########################################################################################
MEDIA = {
    "air": {
        "name": "Air (20C, 1 atm)",
        "rho": 1.225e-6,                # density in g/mm³  (1.225 kg/m³)
        "c": 343000.0,                    # speed of sound in mm/s  (343 m/s)
        "description": "Standard air at room temperature and atmospheric pressure",
        "kappa": 6.939e-6                 # compressibility
    },
    # Future media can be added here:
    # "water": {
    #     "name": "Water (20°C)",
    #     "rho": 1.0e-9,
    #     "c": 1_480_000.0,
    #     "description": "Fresh water at room temperature",
    # },
}

##########################################################################################
# Object Material Presets
##########################################################################################
MATERIALS = {
    "solid_plastic": {
        "name": "Solid Plastic (PLA/ABS)",
        "rho": 1.250e-3,                  # density in g/mm³  (1250 kg/m³)
        "c": 2_400_000.0,                # speed of sound in mm/s  (2400 m/s)
        "description": "Standard 3D printing filament, solid infill",
        "kappa": 1.389e-10,
    },
    "polystyrene_foam": {
        "name": "Polystyrene Foam (EPS)",
        "rho": 0.025e-3,                 # ~25 kg/m³
        "c": 1_800_000.0,                # ~1800 m/s
        "description": "Expanded polystyrene foam — commonly levitated",
        "kappa": 1.235e-8,
    },
    "water_droplet": {
        "name": "Water Droplet",
        "rho": 1.000e-3,                  # ~1000 kg/m³
        "c": 1_480_000.0,                # ~1480 m/s
        "description": "Liquid water droplet",
        "kappa": 4.566e-10,
    },
    "glass": {
        "name": "Glass (Borosilicate)",
        "rho": 2.230e-3,                  # ~2230 kg/m³
        "c": 5_640_000.0,                # ~5640 m/s
        "description": "Borosilicate laboratory glass",
        "kappa": 1.412e-11,
    },
    "steel": {
        "name": "Steel Bearing",
        "rho": 7.850e-3,                  # ~7850 kg/m³
        "c": 5_960_000.0,                # ~5960 m/s
        "description": "Stainless steel ball bearing",
        "kappa": 3.587e-12,
    },
    "aluminum": {
        "name": "Aluminum (6061)",
        "rho": 2.700e-3,                  # ~2700 kg/m³
        "c": 6_420_000.0,                # ~6420 m/s
        "description": "Aluminum 6061-T6 alloy",
        "kappa": 9.000e-12,
    },
}

##########################################################################################
# Active Configuration
# >>> CHANGE THESE TO SWITCH MATERIAL / MEDIUM <<<
##########################################################################################
ACTIVE_MEDIUM = "air"
ACTIVE_MATERIAL = "polystyrene_foam"
DEFAULT_SCALE = 0.04

##########################################################################################
# Acoustic Source Parameters
##########################################################################################
FREQUENCY = 40000.0          # Transducer frequency in Hz
AMPLITUDE = 4242.0           # Pressure amplitude scale factor

##########################################################################################
# Helper Functions
##########################################################################################

def get_medium(key=None):
    """Return the active (or specified) medium properties dict."""
    med = MEDIA[key or ACTIVE_MEDIUM]
    # Add derived properties
    result = dict(med)
    result["kappa"] = 1.0 / (med["rho"] * med["c"] ** 2)   # compressibility
    result["omega"] = 2.0 * np.pi * FREQUENCY               # angular frequency
    result["k"] = result["omega"] / med["c"]                 # wavenumber
    result["wavelength"] = med["c"] / FREQUENCY               # wavelength in mm
    return result


def get_material(key=None):
    """Return the active (or specified) material properties dict."""
    mat = MATERIALS[key or ACTIVE_MATERIAL]
    result = dict(mat)
    result["kappa"] = 1.0 / (mat["rho"] * mat["c"] ** 2)    # compressibility
    return result


def get_contrast_factors(material_key=None, medium_key=None):
    """
    Compute Gorkov contrast factors f1, f2 for a material/medium pair.
    
    f1 = 1 - κ_p / κ_0   (monopole — compressibility contrast)
    f2 = 2(ρ_p - ρ_0) / (2ρ_p + ρ_0)   (dipole — density contrast)
    
    Returns:
        (f1, f2) tuple of floats
    """
    mat = get_material(material_key)
    med = get_medium(medium_key)

    kappa_p = mat["kappa"]
    kappa_0 = med["kappa"]
    rho_p = mat["rho"]
    rho_0 = med["rho"]

    f1 = 1.0 - kappa_p / kappa_0           # monopole (compressibility)
    f2 = 2.0 * (rho_p - rho_0) / (2.0 * rho_p + rho_0)  # dipole (density)
    return f1, f2


def print_active_config():
    """Print the current active material/medium configuration."""
    med = get_medium()
    mat = get_material()
    f1, f2 = get_contrast_factors()

    print("=" * 60)
    print("  ACOUSTIC MATERIAL CONFIGURATION")
    print("=" * 60)
    print(f"  Medium:    {med['name']}")
    print(f"    rho_0 = {med['rho']:.3e} g/mm^3  ({med['rho']*1e12:.3f} kg/m^3)")
    print(f"    c_0   = {med['c']:.0f} mm/s  ({med['c']/1000:.0f} m/s)")
    print(f"    kappa = {med['kappa']:.3e} mm^2/g")
    print(f"    lambda= {med['wavelength']:.3f} mm")
    print(f"    k     = {med['k']:.4f} rad/mm")
    print()
    print(f"  Material:  {mat['name']}")
    print(f"    rho_p = {mat['rho']:.3e} g/mm^3  ({mat['rho']*1e12:.3f} kg/m^3)")
    print(f"    c_p   = {mat['c']:.0f} mm/s  ({mat['c']/1000:.0f} m/s)")
    print(f"    kappa = {mat['kappa']:.3e} mm^2/g")
    print()
    print(f"  Gorkov Contrast Factors:")
    print(f"    f1 = {f1:.6f}  (monopole / compressibility)")
    print(f"    f2 = {f2:.6f}  (dipole / density)")
    print("=" * 60)


# --- Run as standalone to see current config ---
if __name__ == "__main__":
    print_active_config()

    print("\nAll available presets:")
    for key, mat in MATERIALS.items():
        f1, f2 = get_contrast_factors(material_key=key)
        print(f"  {key:20s}  f1={f1:.6f}  f2={f2:.6f}  — {mat['description']}")
