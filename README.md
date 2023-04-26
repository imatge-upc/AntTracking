# AntTracking
Tracking Ants in a lab table



This code is intended to detect and track ants over a white table.

- ant_detection.py: use background substraction to detect the ants. Write the results in a text file using the MOTChallenge format
- plot_rectangles_video.py: From a MOTChallenge format tracking file and a video file, plot the detections at each frame. The color represents the trackID.
- tracking_utils.py: several functions for tracking
- plot_tracks.py: Plot a graphical representation of the tracks along the frames. 
