import numpy as np
# This function will spawn a target outside of a central circular area
'''
inputs:
    radius - central radius to be excluded 
    maxX - maximum of X value
    maxY - maximum of Y value
    outerBuffer - sets a 5% edge area where targets won't spawn

    **radius must be less than maxX and maxY

outputs:
    x, y coordinates of a random location outside of the circle
'''
def targetSpawn(radius, maxX, maxY, outerBuffer=True):
    assert radius < maxX and radius < maxY, "Radius of the circle is too large! It must be less than the maximum X and Y values"
    if outerBuffer:
        maxX *= .95
        maxY *= .95
        
    x = np.random.uniform(-maxX, maxX)
    thresh = np.sqrt( abs(radius**2-x**2) ) if abs(x) < radius else 0
    y = np.random.uniform(thresh, maxY) * np.random.choice([-1, 1])

    return x, y




# Uncomment below to test and visualize random spawning
'''
import matplotlib.pyplot as plt

radius = 0.4
maxX=1
maxY=1
listPts = [targetSpawn(radius, maxX, maxY) for _ in range(100)]


plt.rcParams["figure.figsize"] = (20,20)
fig, axes = plt.subplots()

axes.set_aspect(1)
plt.scatter(*zip(*listPts))
axes.add_artist(plt.Circle((0, 0), radius, fill = False ))

plt.show()
'''