##########################################################################################
 # SimAcousticField.py
##########################################################################################
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import time
##########################################################################################
 # Helper functions
##########################################################################################
def arrayprint(arr):
    for i in range(len(arr)):
        print(f"Source {i+1}: {arr[i]}")
    print("\n")

##########################################################################################
 # Import Acoustic Source Locations from text file
##########################################################################################
rawsrc = np.loadtxt('AcousticFieldModeling/srcarray.txt')
srcarray = rawsrc.reshape(-1, 3)
arrayprint(srcarray)

##########################################################################################
 # Create Acoustic Pressure Simulation Space
##########################################################################################
simspace = np.zeros((81, 81, 81)) # 80x80x80 mm^3 space
# Define the linespace for x, y, z coordinates
x = np.linspace(-40, 40, 81) # from -40 to 40 mm
y = np.linspace(-40, 40, 81) # from -40 to 40 mm
z = np.linspace(-40, 40, 81) # from -40 to 40 mm
# Create a meshgrid for the simulation space
X, Y, Z = np.meshgrid(x, y, z)

##########################################################################################
 # Define a function for calculating the acoustic pressure field at a single point
##########################################################################################
def acoustic_pressure_field(point_coords, sources):
    """
    Calculates the acoustic pressure field using coherent wave addition.
    P = sum( (A/r) * exp(i * (k*r + phi)) )
    """
    c = 343000.0 # Speed of sound in mm/s
    f = 40000.0   # Frequency in Hz
    wl = c / f
    k = 2 * np.pi / wl
    A = 4242.0 # Scale factor
    
    # Distance to all sources
    r = np.linalg.norm(sources - point_coords, axis=1)
    r = np.clip(r, 1e-3, None)
    
    # Complex fields sum
    angle = k * r # Assuming phase=0 for all
    complex_fields = (A / r) * (np.cos(angle) + 1j * np.sin(angle))
    total_field = np.sum(complex_fields)
    
    return np.abs(total_field)

##########################################################################################
 # Evaluate Pressure Field
##########################################################################################
start = time.time()
for i in range(simspace.shape[0]):
    for j in range(simspace.shape[1]):
        for k in range(simspace.shape[2]):
            simspace[i, j, k] = acoustic_pressure_field(np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]]), srcarray)
    print("YZ-Plane, x = ", x[i], "mm evaluated.")
end = time.time()
print("Simulation completed in", end - start, "seconds.")

##########################################################################################
 # Plot XY Slice of the Acoustic Pressure Field
##########################################################################################

plt.figure(figsize=(8, 6))
plt.imshow(simspace[:, :, simspace.shape[2]//2], extent=(-40, 40, -40, 40), origin='lower', cmap='viridis')
plt.colorbar(label='RMS Pressure (Pa)')
plt.title('RMS Acoustic Pressure Field')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.xlim(-25, 25)
plt.ylim(-30, 30)
# change color bar limits to enhance contrast
plt.clim(0, 4000)
# plt.grid()
plt.show()
