#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(net_image, send_dict):
    global conn

    # image
    if net_image != None:
        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        conn.sendall(net_image_bytes)
    
    # dict
    dict_str = json.dumps(send_dict)

    conn.sendall(len(dict_str).to_bytes(4, 'little'))
    conn.sendall(dict_str.encode("utf-8"))

def receive():
    msg = read()

    width = msg["resolution_x"]
    height = msg["resolution_y"]
    msg["do_training"] = bool(msg["do_training"])
    msg["keep_alive"] = bool(msg["keep_alive"])

    if width != 0 and height != 0:
        try:
            world_view_transform = torch.reshape(torch.tensor(msg["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(msg["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            timestep = msg["timestep"] if "timestep" in msg else None
            custom_cam = MiniCam(width, height, msg["fov_y"], msg["fov_x"], msg["z_near"], msg["z_far"], world_view_transform, full_proj_transform, timestep)
        except Exception as e:
            print("")
            print(e)
            traceback.print_exc()
            raise e
        return custom_cam, msg
    else:
        return None, msg