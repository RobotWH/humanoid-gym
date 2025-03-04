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
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
import signal
import sys

class KeyboardThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.target_vel_x = 0.0
        self.target_vel_y = 0.0
        self.target_ang_vel_yaw = 0.0
        self.hallo = False
        self.shake_hands = False
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
                elif key.char == 'h': 
                    self.hallo =  True
                elif key.char == 'j': 
                    self.shake_hands =  True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            with self.lock:
                if key.char == 'e' or key.char == 'q':
                    self.target_ang_vel_yaw = 0.0  # 松开 e 或 q 时重置角速度
                if key.char == 'h':
                    self.hallo = False  
                if key.char == 'j':
                    self.shake_hands = False  
        except AttributeError:
            pass

    def run(self):
        self.listener.start()
        while self.running:
            time.sleep(0.001)  # ✅ 释放CPU资源

    def get_velocity(self):
        with self.lock:
            return self.target_vel_x, self.target_vel_y,self.target_ang_vel_yaw

    def get_arms_cmd(self):
        with self.lock:
            return self.hallo, self.shake_hands
        
    def stop(self):
        self.running = False
        self.listener.stop()
        self.join()

joints_name = [
        "neck_yaw_joint",
        "neck_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_arm_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_arm_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "left_leg_roll_joint",
        "left_leg_yaw_joint",
        "left_leg_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_leg_roll_joint",
        "right_leg_yaw_joint",
        "right_leg_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint"
]

joints_dict = dict.fromkeys(joints_name, 0)

play_bag = False

rl_q = 0
rl_v = 0
rl_kp = 0
rl_kd = 0 
rl_tor = 0

my_rl= False

def signal_handler(sig, frame):
    rospy.signal_shutdown("User requested shutdown")
    sys.exit(0)  

def joint_state_callback(msg):
    global play_bag
    play_bag = True
    # print(f"play_bag222:{play_bag}")
    for name, pos in zip(msg.name, msg.position):
        if name in joints_dict:
            joints_dict[name] = pos
        # rospy.loginfo("  %-15s: %.3f rad", name, pos)

def bag_state_callback(msg):
    global play_bag
    play_bag = msg.data
    # print(f"play_bag:{play_bag}")

def my_rl_state_callback(msg):
    global my_rl
    my_rl = msg.data
    # print(f"play_bag:{play_bag}")

def rl_result_callback(msg):
    global my_rl,rl_q,rl_v,rl_kp,rl_kd,rl_tor
    
    rl_q = msg.data[18:30]
    rl_q = np.array(rl_q, dtype=np.float64)
    rl_v = msg.data[48:60] 
    rl_v = np.array(rl_v, dtype=np.float64)
    rl_kp = msg.data[78:90] 
    rl_kp = np.array(rl_kp, dtype=np.float64)
    rl_kd = msg.data[108:120] 
    rl_kd = np.array(rl_kd, dtype=np.float64)
    rl_tor = msg.data[138:150] 
    rl_tor = np.array(rl_tor, dtype=np.float64)
    # print(f"rl_tor:{rl_tor}")
    # my_rl= True

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

def euler_to_quaternion_array(euler_angles):
    # 输入参数为欧拉角数组，单位弧度，顺序为 [roll_x, pitch_y, yaw_z]
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    
    # 计算各轴旋转的半角三角函数值
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # 四元数分量计算（按ZYX旋转顺序组合）
    w = cr * cp * cy + sr * sp * sy  # 实部
    x = sr * cp * cy - cr * sp * sy  # X虚部
    y = cr * sp * cy + sr * cp * sy  # Y虚部
    z = cr * cp * sy - sr * sp * cy  # Z虚部
    
    return np.array([x, y, z, w])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    # print(f"q.shape:{q.shape}")
    np.set_printoptions(suppress=True, precision=3)
    # print(f"right_arm_current_q:{q[-19:-12]}")
    # print(f"left_arm_current_q:{q[-26:-19]}")
    dq = data.qvel.astype(np.double)
    tq = data.actuator_force.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    acc = data.sensor('linear-acceleration').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec,tq,acc)

def xbot_state_pub(cfg,cmd_pub,q,dq,quat,omega,tq,acc):
    msg = Float64MultiArray()
    msg.data = [0] * (36 * 3 + 4 + 3 * 2)
    msg.data[24:36] = q[-cfg.env.num_actions:]
    msg.data[60:72] = dq[-cfg.env.num_actions:]
    msg.data[96:108] = tq[-cfg.env.num_actions:]
    euang = quaternion_to_euler_array(quat)
    # print(f"euang:{euang}")
    euang[1:3] = -euang[1:3]
    quat_tmp = euler_to_quaternion_array(euang)
    # print(f"quat_tmp:{quat_tmp}")
    quat_t=np.zeros((4), dtype=np.double)
    quat_t[0] = quat_tmp[3]
    quat_t[1] = quat_tmp[0]
    quat_t[2] = quat_tmp[1]
    quat_t[3] = quat_tmp[2]
    # print(f"quat_t:{quat_t}")
    msg.data[108:112] = quat_t
    euang_t = quaternion_to_euler_array(quat_t)
    # print(f"euang_t:{euang_t}")

    omega_copy = np.copy(omega)
    omega_copy[1:3] = -omega_copy[1:3]
    msg.data[112:115] = omega_copy
    msg.data[115:118] = acc
    # print(f"omega:{omega}")
    cmd_pub.publish(msg)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    # print(f"target_q.shape:{target_q.shape},q:{q.shape}")
    # print(f"target_dq.shape:{target_dq.shape},dq:{dq.shape}")

    return (target_q - q) * kp + (target_dq - dq) * kd

def pd_control_tor(target_q, q, kp, target_dq, dq, kd,t):
    '''Calculates torques from position commands
    '''
    # print(f"target_q.shape:{target_q.shape},q:{q.shape}")
    # print(f"target_dq.shape:{target_dq.shape},dq:{dq.shape}")

    return (target_q - q) * kp + (target_dq - dq) * kd +t

def get_joint_names(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # 直接遍历关节索引获取名称
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    
    # 过滤空值（未命名的关节）
    joint_names = [name for name in joint_names if name]
    
    return joint_names

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    global my_rl,rl_q,rl_v,rl_kp,rl_kd,rl_tor,play_bag
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('xbot_mujoco_simulator', anonymous=True)
    while not rospy.is_shutdown():
        rospy.Subscriber("/xbot_joints", JointState, joint_state_callback, queue_size=1)
        rospy.Subscriber("/policy_input", Float64MultiArray, rl_result_callback, queue_size=1)
        rospy.Subscriber("/bag_state", Bool, bag_state_callback)
        rospy.Subscriber("/my_rl_state", Bool, my_rl_state_callback)
        cmd_pub = rospy.Publisher('/controllers/xbot_controller/policy_output', Float64MultiArray, queue_size=1)

        all_joints = cfg.env.num_actions+cfg.sim_config.num_arms_joints
        keyboard_thread = KeyboardThread()
        keyboard_thread.start()
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        # for i in range(model.njnt):
        #     jnt_type = model.jnt_type[i]
        #     jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        #     print(f"关节 {jnt_name} 类型: {jnt_type} (自由度: {model.jnt_dofadr[i]})")
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        viewer = mujoco_viewer.MujocoViewer(model, data)
        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        target_q_myrl = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)
        hist_obs = deque()
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

        count_lowlevel = 0
        move_lowlevel =  0
        hallo_time = 0 
        shake_hand_time = 0
        all_time = 30

        for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
            # Obtain an observation
            q, dq, quat, v, omega, gvec,tq,acc = get_obs(data)
            xbot_state_pub(cfg,cmd_pub,q,dq,quat,omega,tq,acc)
            q_leg = q[-cfg.env.num_actions:]
            dq_leg = dq[-cfg.env.num_actions:]
            
            # 1000hz -> 100hz
            if count_lowlevel % cfg.sim_config.decimation == 0:
                if not my_rl:
                    vel_x,vel_y,ang_vel_yaw = keyboard_thread.get_velocity()
                    hallo,shake_hand = keyboard_thread.get_arms_cmd()
                    print(f"vel_x:{vel_x}, vel_y:{vel_y},ang_vel_yaw:{ang_vel_yaw}")
                    print(f"hallo:{hallo},shake_hand:{shake_hand},play_bag:{play_bag},my_rl:{my_rl}")
                    obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                    eu_ang = quaternion_to_euler_array(quat)
                    # quat_a = euler_to_quaternion_array(eu_ang)
                    # print(f"quat_diff:{quat-quat_a}")
                    eu_ang[eu_ang > math.pi] -= 2 * math.pi
                    if math.sqrt(vel_x*vel_x+vel_y*vel_y+ang_vel_yaw*ang_vel_yaw)<cfg.commands.stand_com_threshold:
                        move_lowlevel = 0
                    obs[0, 0] = math.sin(2 * math.pi * move_lowlevel * cfg.sim_config.dt  / 0.64)
                    obs[0, 1] = math.cos(2 * math.pi * move_lowlevel * cfg.sim_config.dt  / 0.64)
                    obs[0, 2] = vel_x * cfg.normalization.obs_scales.lin_vel
                    obs[0, 3] = vel_y * cfg.normalization.obs_scales.lin_vel
                    obs[0, 4] = ang_vel_yaw * cfg.normalization.obs_scales.ang_vel
                    obs[0, 5:17] = q_leg * cfg.normalization.obs_scales.dof_pos
                    obs[0, 17:29] = dq_leg * cfg.normalization.obs_scales.dof_vel
                    obs[0, 29:41] = action
                    obs[0, 41:44] = omega
                    obs[0, 44:47] = eu_ang
                    # print(f"obs:{eu_ang}")
                    obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                    hist_obs.append(obs)
                    hist_obs.popleft()

                    policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                    for i in range(cfg.env.frame_stack):
                        policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
                    action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                    action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

                    target_q = action * cfg.control.action_scale

                    left_arm_joints = np.array([-0.2, 0.0, 0.0, 0.15, 0.0, 0.20, 0.0])
                    right_arm_joints = np.array([0.2, 0.0, 0.0, -0.15, 0.0, -0.20, 0.0])
                    hallo_joints = np.array([2.6, 0.0, 0.0, 0.0, -1.60, 0.0, 0.0])
                    shake_hand_joints = np.array([0.5, 0.0, 0.0, -0.5, -0.00, -0.6, 0.0])
                    bag_left_arm = np.array([   joints_dict['left_shoulder_pitch_joint'],
                                                joints_dict['left_shoulder_roll_joint'],
                                                joints_dict['left_arm_yaw_joint'],
                                                joints_dict['left_elbow_pitch_joint'],
                                                joints_dict['left_elbow_yaw_joint'],
                                                joints_dict['left_wrist_roll_joint'],
                                                joints_dict['left_wrist_yaw_joint']])
                    bag_right_arm = np.array([  joints_dict['right_shoulder_pitch_joint'],
                                                joints_dict['right_shoulder_roll_joint'],
                                                joints_dict['right_arm_yaw_joint'],
                                                joints_dict['right_elbow_pitch_joint'],
                                                joints_dict['right_elbow_yaw_joint'],
                                                joints_dict['right_wrist_roll_joint'],
                                                joints_dict['right_wrist_yaw_joint']])
                    # print(f"bag_left_arm:{bag_left_arm}")
                    # print(f"play_bag:{play_bag}")
                    
                    if math.sqrt(vel_x*vel_x+vel_y*vel_y+ang_vel_yaw*ang_vel_yaw)>cfg.commands.stand_com_threshold:
                        if obs[0,0] >= 0 :
                            left_arm_joints = np.fabs(obs[0, 0]) * left_arm_joints
                            right_arm_joints = right_arm_joints *0
                        else:
                            right_arm_joints = np.fabs(obs[0, 0]) * right_arm_joints
                            left_arm_joints = left_arm_joints * 0
                    elif not play_bag :
                        if hallo :
                            if hallo_time < all_time:
                                hallo_time += 1
                            left_arm_joints  = left_arm_joints * 0 
                            right_arm_joints = hallo_joints * (hallo_time/all_time)
                        elif shake_hand:
                            if shake_hand_time < all_time:
                                shake_hand_time += 1
                            left_arm_joints  = left_arm_joints * 0 
                            right_arm_joints = shake_hand_joints * (shake_hand_time/all_time) 
                        else:
                            if hallo_time>0:
                                hallo_time -= 0.5
                                left_arm_joints  = left_arm_joints * 0 
                                right_arm_joints =  hallo_joints * (hallo_time/all_time)
                            elif shake_hand_time>0:
                                shake_hand_time -= 0.5
                                left_arm_joints  = left_arm_joints * 0 
                                right_arm_joints =  shake_hand_joints * (shake_hand_time/all_time) 
                            else:
                                left_arm_joints  = left_arm_joints * 0 
                                right_arm_joints = right_arm_joints * 0 
                    elif play_bag:
                        left_arm_joints = bag_left_arm
                        right_arm_joints = bag_right_arm
                    # print(f"hallo_time:{hallo_time},shake_hand_time:{shake_hand_time}")
                    # print(f"left_arm_joints:{left_arm_joints}")
                    target_q = np.concatenate([right_arm_joints, target_q])  # 默认沿axis=0拼接
                    target_q = np.concatenate([left_arm_joints, target_q])  # 默认沿axis=0拼接
                    # print(f"target_q:{target_q}")
                else:
                    left_arm_joints = np.zeros((7), dtype=np.double)
                    right_arm_joints = np.zeros((7), dtype=np.double)
                    target_q_myrl = rl_q
                    print(f"rl_q:{rl_q},my_rl:{my_rl}")
                    target_q_myrl = np.concatenate([right_arm_joints, target_q_myrl])  # 默认沿axis=0拼接
                    target_q_myrl = np.concatenate([left_arm_joints, target_q_myrl])  # 默认沿axis=0拼接
                    # print(f"target_q:{target_q[-12:]}")
                    # print(f"target_q_myrl:{target_q_myrl[-12:]}")
                    # print(f"target_diff:{target_q[-12:]-target_q_myrl[-12:]}")



            
            target_dq = np.zeros((all_joints), dtype=np.double)
            # tau = np.zeros((all_joints), dtype=np.double)
            # Generate PD control
            if not my_rl:
                tau = pd_control(target_q[-all_joints:], q[-all_joints:], cfg.robot_config.kps[-all_joints:],
                                target_dq[-all_joints:], dq[-all_joints:], cfg.robot_config.kds[-all_joints:])  # Calc torques
            else:
                # arm_tor = np.zeros((cfg.sim_config.num_arms_joints), dtype=np.double)
                # rl_all_tor = np.concatenate([arm_tor, rl_tor])
                # print(f"rl_all_tor:{rl_all_tor}")
                tau = pd_control(target_q_myrl[-all_joints:], q[-all_joints:], cfg.robot_config.kps[-all_joints:],
                                target_dq[-all_joints:], dq[-all_joints:], cfg.robot_config.kds[-all_joints:])
                # print(f"tau_myrl:{tau_myrl}")
        
            tau_limit = 200. * np.ones(all_joints, dtype=np.double)
            tau = np.clip(tau, -tau_limit, tau_limit)  # Clamp torques
            # print(f"tau:{tau}")

            data.ctrl = tau
            mujoco.mj_step(model, data)
            viewer.render()
            count_lowlevel += 1
            move_lowlevel += 1
            
            # play_bag = False
            rospy.sleep(0.001)  # 控制ROS处理频率

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--arms', action='store_true', help='arm or no arm')
    args = parser.parse_args()
    # print(f"args:{args}")

    class Sim2simCfg(XBotLNoArmsCfg):
        
        class sim_config:
            if args.arms:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-arms.xml'
                num_arms_joints = 14
                print("--------WITH ARMS-----")
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'
                num_arms_joints = 0
                print("--------NO ARMS------")

            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        if args.arms:
            class robot_config:
      
                kps = np.array([200, 200, 200, 200, 200, 200, 200,
                                200, 200, 200, 200, 200, 200, 200,
                                200, 200, 350, 350, 15,  15, 
                                200, 200, 350, 350, 15,  15       ], dtype=np.double)
                myrl_kps = np.array([200, 200, 200, 200, 200, 200, 200,
                                200, 200, 200, 200, 200, 200, 200,
                                200, 200, 350, 350, 0,  0, 
                                200, 200, 350, 350, 0,  0       ], dtype=np.double)

                kds = np.array([10,10,25,10,15, 5 ,10,
                                10,10,25,10,15, 5 ,10, 
                                10,10,10,10,10, 10, 
                                10,10,10,10,10, 10    ], dtype=np.double)
                myrl_kds = np.array([10,10,25,10,15, 5 ,10,
                                10,10,25,10,15, 5 ,10, 
                                10,10,10,10,2, 2, 
                                10,10,10,10,2, 2    ], dtype=np.double)
        else:
            class robot_config:
                kps = np.array([200, 200, 350, 350, 15, 15, 
                                200, 200, 350, 350, 15, 15], dtype=np.double)
                myrl_kps = np.array([   200, 200, 350, 350, 0, 0, 
                                        200, 200, 350, 350, 0, 0], dtype=np.double)
                kds = np.array([10, 10, 10, 10, 10, 10, 
                                10, 10, 10, 10, 10,10], dtype=np.double)
                myrl_kds = np.array([10, 10, 10, 10, 2, 2, 
                                10, 10, 10, 10, 2, 2], dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
