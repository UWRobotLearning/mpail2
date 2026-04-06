# Installation

> Tested on **Ubuntu 22.04** and **Ubuntu 20.04** with **Python 3.10**

## TL;DR

| Setup | Before installing `mpail2` |
| --- | --- |
| 🧠 Core algorithms | None |
| 🎮 Gym / MuJoCo | None |
| 🧪 IsaacLab / Isaac Sim | [Install Isaac Sim / IsaacLab](#isaaclab--isaac-sim) |
| 🦾 Franka real | Set up the [realtime kernel](https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html), then install `mpail2` with `".[franka]"` |
| 🤖 Kinova real | Install [ROS 2](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) and [`ros2_kortex`](https://github.com/Kinovarobotics/ros2_kortex) |

## Default Install

For the default install, use:

```bash
conda create -n mpail2 python=3.10
conda activate mpail2
pip install --upgrade pip
pip install -e .
```

This installs the base Python dependencies for `mpail2`, including Gym /
MuJoCo support.

IsaacLab and the real-robot environments require extra setup:

- IsaacLab / Isaac Sim: install Isaac Sim / IsaacLab first, then install `mpail2`
- Franka real robot: install `mpail2` with `".[franka]"`
- Kinova real robot: install ROS 2 / `ros2_kortex`

## Gym / MuJoCo

Gym / MuJoCo support is included in the default install.

## IsaacLab / Isaac Sim

If you plan to use IsaacLab environments, install Isaac Sim / IsaacLab first and create a conda environment through their provided script. Install mpail2, then finally install the isaaclab dependencies:

0. Create Isaac Lab conda environment via provided script. Skip if you have an existing environment you want to use. `isaacsim` may not be detected if you do not use the IsaacLab setup script.

    ```bash
    cd IsaacLab
    ./isaaclab.sh -c mp2isaac
    ```

1. Activate the Python environment.
    ```bash
    conda activate mp2isaac
    ```
2. `cd` to the `mpail2` source. Install `mpail2` into the conda env you just created.
    ```bash
    cd <mpail2> # wherever you cloned mpail2
    pip install -e .
    ```
3. Install IsaacLab / Isaac Sim.

    <details>
    <summary><strong>with pip (Ubuntu 22.04+)</strong></summary>

    Follow the official Isaac Lab installation guide for your platform and release. In this repo, keep the environment on Python 3.10. One pip-based path is:

    ```bash
    pip install flatdict==4.0.1 --no-build-isolation  # workaround for IsaacLab flatdict build issue
    pip install isaaclab[isaacsim,all] --extra-index-url https://pypi.nvidia.com
    ```
    </details>

    <details>
    <summary><strong>from source</strong></summary>

    Follow the official Isaac Lab installation guide for your platform and release.    

    ```bash
    cd IsaacLab
    conda activate mp2isaac
    ./isaaclab.sh -i # install isaaclab deps
    ```

    </details>

4. Verify the IsaacLab install works.
    ```bash
    python -c "from isaaclab.app import AppLauncher; print('IsaacLab import OK')"
    ```

Official guides:

- [Isaac Lab quickstart](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html)
- [Isaac Lab local installation and driver requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#driver-requirements)
## 🦾 Franka (Real Robot) Setup

### Overview

You will:
1. Enable real-time control
2. Install Franka control dependencies
3. Start a robot-side server
4. Connect from your code

### Step 0. Network + Robot Access

- Your machine must be on the same network as the robot (typically `172.16.0.x`)
- You need:
  - Robot IP (e.g. `172.16.0.2`)
  - FCI enabled (via Franka web interface)

Test connectivity:
```bash
ping <robot_ip>
```

### Step 1. Real-Time Kernel

Check if already installed:

```bash
uname -a  # you should see `PREEMPT_RT`
```

If not, install it:
[https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html](https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html)

### Step 2. Real-Time Permissions

```bash
# 1. Create realtime group and add your user
sudo addgroup realtime
sudo usermod -a -G realtime $(whoami)

# 2. Add the following lines to /etc/security/limits.conf
@realtime soft rtprio 99
@realtime hard rtprio 99
@realtime soft memlock 102400
@realtime hard memlock 102400

# 3. Log out and back in, then verify
ulimit -r   # should be 99
```

### Step 3. Install Franka Control Stack

Install `mpail2` with the Franka extra:

```bash
pip install -e ".[franka]"
```

### Step 4. Start Robot Server

Run this on the machine connected to the robot:

```bash
python -m mpail2.envs.real.franka.network.server \
    --robot_ip <robot_ip> \
    --bind_address "tcp://*:5555"
```

### Step 5. Verify Control

From another terminal (or another machine):

```python
from mpail2.envs.real.franka.network import FrankaClient
import numpy as np

with FrankaClient("tcp://<server_ip>:5555", dynamics_factor=0.05) as client:
    client.reset()
    # action = [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
    # first 6 dims are Cartesian deltas; gripper is in [-1, 1]
    action = np.zeros(7, dtype=np.float32)
    action[6] = 1.0  # open the gripper
    client.step(action, blocking=True)
```

The vendored ``FrankaClient`` in ``mpail2`` supports only ``cartesian_delta``,
which is the mode used by MPAIL.

If the robot moves, setup is working.

## 🤖 Kinova (ROS 2) Setup

This setup uses **ROS 2 Humble + `ros2_kortex`**.

### Overview

You will:

1. Install ROS 2
2. Build Kinova drivers
3. Connect to the robot
4. Verify ROS topics

### Step 0. Network Setup

* Robot and PC must be on the same network
* Default IP is typically:

    ```
    192.168.1.10
    ```

    Test:

    ```bash
    ping 192.168.1.10
    ```

### Step 1. Install ROS 2 Humble

```bash
# 1. Setup locale (required)
sudo apt update
sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. Add ROS 2 apt repository
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu \
$(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 3. Install ROS 2
sudo apt update
sudo apt install ros-humble-desktop ros-dev-tools

# 4. Add to `.bashrc`
source /opt/ros/humble/setup.bash

# 5. Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### Step 2. Install `ros2_kortex`

```bash
# 1. Clone the repo
cd ~/ros2_ws/src
git clone https://github.com/Kinovarobotics/ros2_kortex.git


# 2. Install dependencies
cd ~/ros2_ws/
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 3. Build
colcon build

# 4. Source
source install/setup.bash
```

### Step 3. Connect to Robot

```bash
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10
```

### Step 4. Verify

```bash
ros2 topic list  # list topics
ros2 topic echo /joint_states  # check joint_states
```

If data is streaming -> setup is working.
