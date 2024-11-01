# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import XBotLCfg,XBotLNoArmsCfg
import torch
from pynput import keyboard
import threading,time


class KeyboardThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.target_vel_x = 0.0
        self.target_vel_y = 0.0
        self.target_ang_vel_yaw = 0.0
        self.running = True

        # 初始化监听器
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )

    def on_press(self, key):
        try:
            with self.lock:
                if  key.char == 'w':
                    self.target_vel_x = min(self.target_vel_x + 0.1, 1)  # 前进速度
                    # print(f"self.target_vel_x :{self.target_vel_x}")
                elif key.char == 's':
                    self.target_vel_x = max(self.target_vel_x - 0.1, -1)  # 后退速度
                elif key.char == 'a':
                    self.target_vel_y = min(self.target_vel_y + 0.1, 0.5)   # 左平移
                elif key.char == 'd':
                    self.target_vel_y = max(self.target_vel_y - 0.1, -0.5)  # 右平移
                elif key.char == 'e':
                    self.target_ang_vel_yaw = -0.5  # 左转速度
                elif key.char == 'q':
                    self.target_ang_vel_yaw = 0.5  # 右转速度
                elif key.char == 'r': 
                    self.target_vel_x = 0.0
                    self.target_vel_y = 0.0
                    self.target_ang_vel_yaw = 0.0
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            with self.lock:
                if key.char == 'e' or key.char == 'q':
                    self.target_ang_vel_yaw = 0.0  # 松开 e 或 q 时重置角速度
        except AttributeError:
            pass

    def run(self):
        self.listener.start()
        while self.running:
            time.sleep(0.001)  # ✅ 释放CPU资源

    def get_velocity(self):
        with self.lock:
            return self.target_vel_x, self.target_vel_y,self.target_ang_vel_yaw

    def stop(self):
        self.running = False
        self.listener.stop()
        self.join()

class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0

def  _get_phase(cfg):
    cycle_time = cfg.rewards.cycle_time

    # print(f"self.gait_start:{self.gait_start}")
    if self.cfg.commands.sw_switch:
        stand_command = (torch.norm(self.commands[:, :2], dim=1) <= self.cfg.commands.stand_com_threshold)
        self.phase_length_buf[stand_command] = 0 # set this as 0 for which env is standing
        # self.gait_start is rand 0 or 0.5
        phase = (self.phase_length_buf * self.dt / cycle_time ) * (~stand_command)
        # print(f"phase:{phase}")
    else:
        phase = self.episode_length_buf * self.dt / cycle_time 
        # print(f"phase:{phase}")
        return phase


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    keyboard_thread = KeyboardThread()
    keyboard_thread.start()
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    move_lowlevel =  0


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation

        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]
        
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            vel_x,vel_y,ang_vel_yaw = keyboard_thread.get_velocity()
            print(f"vel_x:{vel_x}, vel_y:{vel_y},ang_vel_yaw:{ang_vel_yaw}")
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            if math.sqrt(vel_x*vel_x+vel_y*vel_y)<cfg.commands.stand_com_threshold:
                move_lowlevel = 0
            obs[0, 0] = math.sin(2 * math.pi * move_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * move_lowlevel * cfg.sim_config.dt  / 0.64)
            obs[0, 2] = vel_x * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = vel_y * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = ang_vel_yaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
        move_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(XBotLNoArmsCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
