Processing HDR imaging with event and colour cameras
Basic guideline for processing camera images captured from Rosbags in order to detect features for comparing HDR imaging.

Setup code and data folders. Keep the data separate to the code.
Setup Git repository
Setup coding and IDE environment (Python + VSCODE). You will likely need to install packages as you progress through. Make sure to keep a track of all packages using requirements.txt
Create configuration file for running code. This file contains necessary parameters which your script will need (path to data directory, topic names, calibration board specifications). You may also want to consider passing arguments (look into Python parser)
Images and events are stored in Rosbags and camera parameters are stored as YAML files. You will need to write functions to extract that information. Reading from Rosbags is pretty slow because it reads sequentially. You can speedup if reading if you use multiple threads (its a bit complex through). Alternative methods are to stream the Rosbag using "rosbag play ...".
For the event data, we need to convert the events into images. Events messages contain event arrays, each event inside of the array is defined as (x, y, time, polarity). You will need to collect events within a given time span and in order to create an image (start with 1 s). Look into event histograms.
Now you should have event images and colour images. You will need to undistort images using the camera parameters.
Now we need to apply a feature detector to images. Look into SIFT
Filter detected features based on size
Plot number of detected features vs number of actual features.
