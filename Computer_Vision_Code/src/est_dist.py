#contains the functions and algorithms that estimate the distance between the user's eyes and the screen
#separating this logic from main.py helps with readability and organiztion
import numpy as np

def estimate_distance(left_eye, right_eye):
    #calcualte center of each eye
    left_center = np.mean(left_eye, axis=1)
    right_center = np.mean(right_eye, axis=1)

    #distance between the 2 centers
    dist = np.linalg.norm(left_center - right_center)

    return dist