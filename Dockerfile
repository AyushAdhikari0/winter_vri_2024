FROM osrf/ros:noetic-desktop-full

# Change the default shell to Bash
SHELL [ "/bin/bash" , "-c" ]

# Install Python, pip, and other necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

RUN apt-get update && apt install libgl1 -y \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Copy the current directory contents into /code
COPY . /code

WORKDIR /code

RUN pip install --upgrade pip

RUN apt-get update && apt install python3-tk -y && rm -r /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Source the ROS setup script and start a bash shell
ENTRYPOINT ["bash", "-c", "echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc && source ~/.bashrc && bash"]