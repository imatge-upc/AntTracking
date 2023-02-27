import numpy as np

# https://stackoverflow.com/questions/7570808/how-do-i-calculate-the-difference-of-two-angle-measures/30887154
def angular_error(a1,a2):
    phi = np.abs(a1 - a2) % 360   # This is either the distance or 360 - distance
    dist = 360 - phi if phi > 180 else phi
    return dist


