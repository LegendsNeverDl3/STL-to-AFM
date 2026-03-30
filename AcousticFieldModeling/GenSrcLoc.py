##########################################################################################
 # GenSrcLoc.py
##########################################################################################
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

##########################################################################################
 # Define Acoustic Source Locations
##########################################################################################
def arrayprint(arr):
    for i in range(len(arr)):
        print(f"Source {i+1}: {arr[i]}")
    print("\n")


# Define 2 Initial Source Locations
NUM_SRC = 48
srcarray = [None]*NUM_SRC
P1 = [0.000, 38.498, 19.764]
P2 = [6.896, 34.101, 25.735]
print(srcarray)
# Define an axis of rotation
axis = np.array([0, 1, 0]) # y-axis unit vector
# Angle of rotation in radians
theta = np.pi / 6 # 12 transducers per 360 degrees
# Create a rotation vector
rotation_vector = axis * theta
rotation = R.from_rotvec(rotation_vector)

iterRange = [0, (NUM_SRC//2)//2, NUM_SRC//2, NUM_SRC]
# Apply rotation to P1 point and store in srcarray
sP1 = P1
for i in range(iterRange[0], iterRange[1]):
    if i == iterRange[0]:
        srcarray[i] = np.array(sP1)
    else:
        sP1 = rotation.apply(sP1)
        srcarray[i] = np.round(sP1, decimals=3)
arrayprint(srcarray)
# Apply rotation to P2 point and store in srcarray NUM_SRC//2 to NUM_SRC
sP2 = P2
for i in range(iterRange[1], iterRange[2]):
    if i == iterRange[1]:
        srcarray[i] = np.array(sP2)
    else:
        sP2 = rotation.apply(sP2)
        srcarray[i] = np.round(sP2, decimals=3)
arrayprint(srcarray)
# For the second half of the sources, mirror the first half across the xz plane (y=0)
for i in range(iterRange[2], iterRange[3]):
    srcarray[i] = np.array([srcarray[i - NUM_SRC//2][0], -srcarray[i - NUM_SRC//2][1], srcarray[i - NUM_SRC//2][2]])

arrayprint(srcarray)

srcarray = np.array(srcarray)



# Plot the source locations as scatter points in 3D
if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(srcarray[:, 0], srcarray[:, 1], srcarray[:, 2], c='b', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')         
    ax.set_zlabel('Z axis')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-30, 30)
    ax.set_title('Acoustic Source Locations')
    plt.show()

# Save the source locations to a text file
np.savetxt('AcousticFieldModeling/srcarray.txt', srcarray, fmt='%.3f', comments='')