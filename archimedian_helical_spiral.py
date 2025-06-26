import numpy as np
from matplotlib import pyplot as plt

# Parameters for the Archimedean helical spiral
a = 0        # initial radius
b = 0.01      # radial growth per radian
c = 0.15     # vertical growth per radian
t_max = 5 * np.pi  # length of the curve
num_points = 25  # resolution

# Parameter t
t = np.linspace(0, t_max, num_points)

# Archimedean helix equations
r = a + b * t
x = r * np.cos(t)
y = r * np.sin(t) + 0.7
z = c * t

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Archimedean Helical Spiral')
plt.show()
