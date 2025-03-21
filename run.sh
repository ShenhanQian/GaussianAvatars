#!/bin/bash
source /opt/ros/humble/install/setup.bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <avatar_name>"
    exit 1
fi

AVATAR_NAME=$1

python local_viewer.py --point_path /workspace/avatars/$AVATAR_NAME/point_cloud.ply --sh-degree 0 --radius 0.65 --background-color 1 1 1 --demo-mode
