
## Docker

A docker image based on ROS Noetic can be used to run the python script

To build the image run the following command:
```bash
docker build -t vri:latest .
```

The container will try to display using RQT therefore it requires access to your system display. **This container will only work for X11 display-based systems.** Copy and run the following command to start the container (this is a single command):
```bash 
xhost +local:docker && docker run -it --rm --network host --name vri --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume /tmp/.X11-unix:/tmp/.X11-unix vri && xhost -local:docker
```

**NOTE:** Please make sure you type `exit` to shutdown the container such that `xhost` can remove docker privileges to the X server. If container was not shutdown properly or unsure, please run the following command to remove docker privileges to X server:

```bash
xhost -local:docker
```


## Processing HDR imaging with event and colour cameras

Basic guideline for processing camera images captured from Rosbags in order to detect features for comparing HDR imaging.



1. Setup code and data folders. Keep the data separate to the code.
2. Setup Git repository
3. Setup coding and IDE environment (Python + VSCODE). You will likely need to install packages as you progress through. Make sure to keep a track of all packages using requirements.txt
4. Create configuration file for running code. This file contains necessary parameters which your script will need (path to data directory, topic names, calibration board specifications). You may also want to consider passing arguments (look into Python parser)
5. Images and events are stored in Rosbags and camera parameters are stored as YAML files. You will need to write functions to extract that information. Reading from Rosbags is pretty slow because it reads sequentially. You can speedup if reading if you use multiple threads (its a bit complex through). Alternative methods are to stream the Rosbag using "rosbag play ...".
6. For the event data, we need to convert the events into images. Events messages contain event arrays, each event inside of the array is defined as (x, y, time, polarity). You will need to collect events within a given time span and in order to create an image (start with 1 s). Look into event histograms.
7. Now you should have event images and colour images. You will need to undistort images using the camera parameters.
8. Now we need to apply a feature detector to images. Look into [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) 
9. Filter detected features based on size
10. Plot number of detected features vs number of actual features.