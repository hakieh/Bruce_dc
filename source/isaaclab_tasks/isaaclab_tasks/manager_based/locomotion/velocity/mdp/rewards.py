# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def calculate_sin_cos_product(tensor):
    # 计算每行的 x^2 + y^2 + z^2
    squared_sum = torch.sum(tensor**2, dim=1)  # 对每行的元素进行平方和
    
    # 计算每行的 cos(theta) 和 sin(theta)
    v_xy = torch.sqrt(tensor[:, 0]**2 + tensor[:, 1]**2)  # x^2 + y^2 的平方根
    v = torch.sqrt(squared_sum)  # x^2 + y^2 + z^2 的平方根
    
    cos_theta = v_xy / v
    sin_theta = torch.abs(tensor[:, 2]) / v  # 使用绝对值来确保 sin(theta) 为正值
    
    # 计算 sin(theta) * cos(theta)
    sin_cos_product = sin_theta * cos_theta
    
    return sin_cos_product

def calculate_xy(tensor):
    # 计算每行的 x^2 + y^2
    squared_sum = torch.sum(tensor**2, dim=1)  # 对每行的元素进行平方和

    v = torch.sqrt(squared_sum)  
    
 
    res = tensor/ v  

    return res


def calculate_xyz(tensor):
    # 计算每行的 x^2 + y^2
    squared_sum = tensor[:, 0]**2 + tensor[:, 1]**2  # 只计算 x 和 y 的平方和
    
    v_xy = torch.sqrt(squared_sum)  # 计算 x^2 + y^2 的平方根（即 x 和 y 的模长）
    
    # 计算单位向量, 只对 x 和 y 进行归一化
    res = tensor[:, :2] / v_xy.unsqueeze(1)  # 归一化 x 和 y 维度
    tensor[:,:2] = res
    tensor[:,2]=0
    return tensor

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]  #单腿接触地面时才有值，选择两条腿中较低的数值作为奖励
    # print(reward,"reward1")
    reward = torch.clamp(reward, max=threshold)
    # reward = torch.clamp(reward, max=threshold)
    over_limit = torch.logical_or(air_time > 0.3, contact_time > 0.3)
    print(over_limit)
    # print(sensor_cfg.name)
    # print(sensor_cfg.body_ids)
    # print(air_time,"airtime")
    # print(contact_time,"contact time")
    any_over_limit = torch.any(over_limit, dim=1)
    # print(any_over_limit,"any_over_limit")
    reward = torch.where(any_over_limit, -reward, reward)
    
    # print(reward,"reward2")
    # no reward for zero command
    # print(reward.shape,"b")
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_contact_new(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    print(contact_sensor.data.net_forces_w.shape)
    # feet_contact_forces = contact_sensor.data.net_forces_w[:, []e
    contact_f = contact_sensor.data.net_forces_w
    contact_f_bool = torch.norm(contact_f,dim=2) > 2.
    

    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    feet_pos_l = asset.data.body_pos_w[:,5,:]
    feet_pos_r = asset.data.body_pos_w[:,10,:]

    
    theta1 = root_pos - feet_pos_l
    theta2 = root_pos - feet_pos_r
    print(theta1,"theta1")
    print(theta2,"theta2")
    l_feet_theta = calculate_sin_cos_product(theta1)
    r_feet_theta = calculate_sin_cos_product(theta2)
    print(l_feet_theta,"l_feet_theta")
    print(r_feet_theta,"r_feet_theta")
    g = 9.8
    m= 4.44
    equ_rl = g*m*l_feet_theta
    equ_rr = g*m*r_feet_theta
    print(equ_rl,"equ_rl")
    print(equ_rr,"equ_rr")
 
    xy_l = calculate_xyz(theta1)
    xy_r = calculate_xyz(theta2)

    print(xy_l,"xy_l")
    print(xy_r,"xy_r")
    force_l = contact_f[:,0,:]
    force_r = contact_f[:,1,:]
    print(force_l,"force_l")
    print(force_r,"force_r")
    dot_product_l = torch.sum(force_l * xy_l, dim=1)
    dot_product_r = torch.sum(force_r * xy_r, dim=1)
    print(dot_product_l,"dot_product_l")
    print(dot_product_r,"dot_product_r")
    l_reward = torch.where(dot_product_l>0.1*equ_rl,1,0)
    r_reward = torch.where(dot_product_r>0.1*equ_rr,1,0)
    l_reward = torch.where(contact_f_bool[:,0],l_reward,0)
    r_reward = torch.where(contact_f_bool[:,1],r_reward,0)
    print(contact_f_bool,"contact_f_bool------------------------")
    # print(contact_f_bool,"contact_f_bool------------------------")
    print(l_reward,"l_reward")
    print(r_reward,"r_reward")
    reward = l_reward + r_reward
    return reward
    # equ_r = g 
    # print("====================")
    # # compute the reward
    # air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # in_contact = contact_time > 0.0
    # in_mode_time = torch.where(in_contact, contact_time, air_time)
    # single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    # reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]  #单腿接触地面时才有值，选择两条腿中较低的数值作为奖励
    # # print(reward,"reward1")
    # reward = torch.clamp(reward, max=threshold)
    # # reward = torch.clamp(reward, max=threshold)
    # over_limit = torch.logical_or(air_time > 0.3, contact_time > 0.3)
    # print(over_limit)
    # # print(sensor_cfg.name)
    # # print(sensor_cfg.body_ids)
    # # print(air_time,"airtime")
    # # print(contact_time,"contact time")
    # any_over_limit = torch.any(over_limit, dim=1)
    # # print(any_over_limit,"any_over_limit")
    # reward = torch.where(any_over_limit, -reward, reward)
    
    # # print(reward,"reward2")
    # # no reward for zero command
    # # print(reward.shape,"b")
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # return reward

def feet_air_time_positive_biped_sum(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact_time_f = contact_sensor.data.current_contact_time[:, [10,5]]
    in_contact = contact_time > 0.0
    air_time_n = air_time + 1e-6
    contact_time_n = contact_time_f + 1e-6
    # print(air_time,"air_time")
    # print(contact_time,"contact_time")
    pre_time = air_time_n/contact_time_n
    # print(pre_time,"pre_time")
    
    l_time = pre_time[:,0:1]
    r_time = pre_time[:,1:2]
    # print(l_time.shape,"l_shape")
    time = torch.where(in_contact[:,0].unsqueeze(1), r_time,l_time )
    # print(in_contact[:,0].unsqueeze(1).shape,"contact")
    # print(time,"time")
    # print(time.shape,"time shape")
    time_small = torch.min(1/time,time)
    # print(time_small)

    # in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    reward = torch.where(single_stance.unsqueeze(-1), time_small, 0.0).squeeze(0)  #单腿接触地面时才有值，选择两条腿中数值和作为奖励
    # print(reward,"reward")
    # reward = torch.exp(reward)-1

    over_limit = torch.logical_or(air_time > 0.3, contact_time > 0.3)
    # print(sensor_cfg.name)
    # print(sensor_cfg.body_ids)

    any_over_limit = torch.any(over_limit, dim=1).unsqueeze(1)
    # print(any_over_limit.shape,"any_over_limit")
    reward = torch.where(any_over_limit, -reward, reward).squeeze(0)
    reward = reward.squeeze(1)
    # print(reward,"reward")
    # no reward for zero command
    # print(reward.shape,"c")
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward



def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)



def ankle_flat(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    ankle_angle = asset.data.joint_pos[:,14:16]
    # reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    reward = torch.sum(torch.abs(ankle_angle-0.5), dim=1)
    return reward


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # out_of_limits = -(
    #     asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    # ).clip(max=0.0)
    # out_of_limits += (
    #     asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    # ).clip(min=0.0)
    # print(asset.data.joint_pos[:, asset_cfg.joint_ids].shape)
    # print(asset.data.joint_pos[:, :4])
    # fl = ( asset.data.joint_pos[:, 0] ).clip(max=0.0)
    # rl = ( asset.data.joint_pos[:, 2] ).clip(max=0.0)
    # fr = -( asset.data.joint_pos[:, 1] ).clip(min=0.0)
    # rr = -( asset.data.joint_pos[:, 3] ).clip(min=0.0)

    # out_of_limits_RR = -(asset.data.joint_pos[:, 6] - 1.0).clip(min=0.0)
    # out_of_limits_RL = -(asset.data.joint_pos[:, 7] - 1.0).clip(min=0.0)

    # out_of_limits_RR_min = (asset.data.joint_pos[:, 6] - 0).clip(max=0.0)
    # out_of_limits_RL_min = (asset.data.joint_pos[:, 7] - 0).clip(max=0.0)

    # calf_limits_RR_min = (-1.5 - asset.data.joint_pos[:, 10] ).clip(max=0.0)
    # calf_limits_RL_min = (-1.5 - asset.data.joint_pos[:, 11] ).clip(max=0.0)



    out_of_limits_FR = -(asset.data.joint_pos[:, 12] - 1).clip(min=0.0) #大于1给惩罚
    out_of_limits_FL = -(asset.data.joint_pos[:, 13] - 1).clip(min=0.0)

    out_of_limits_FR_min = (asset.data.joint_pos[:, 12] + 1).clip(max=0.0)
    out_of_limits_FL_min = (asset.data.joint_pos[:, 13] + 1).clip(max=0.0) #小于-1给惩罚

    out_of_limits_hip_roll_l = (asset.data.joint_pos[:, 8] + 0.1).clip(max=0.0)
    out_of_limits_hip_roll_r = -(asset.data.joint_pos[:, 9] - 0.1).clip(min=0.0)
    # print(asset.data.joint_pos[:, 8],asset.data.joint_pos[:, 9])

    # calf_limits_FR_min = (-1.5 - asset.data.joint_pos[:, 8] ).clip(max=0.0)
    # calf_limits_FL_min = (-1.5 - asset.data.joint_pos[:, 9] ).clip(max=0.0)

    # print(asset.data.joint_pos[:, [6,7]])
    # ooo

    
    # print(fl,rl,fr,rr,"--------")
    # print(asset.data.soft_joint_pos_limits,"=======================")
    # print(asset.data.joint_names,"=======================") ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

    # print(asset_cfg.joint_ids)
    # out_of_limits =  #fl + rl + fr + rr + out_of_limits_RR+ out_of_limits_RL + out_of_limits_RR_min + out_of_limits_RL_min + calf_limits_RR_min + calf_limits_RL_min\
    out_of_limits = out_of_limits_FR + out_of_limits_FL + out_of_limits_FR_min + out_of_limits_FL_min +out_of_limits_hip_roll_l +out_of_limits_hip_roll_r #+ calf_limits_FR_min + calf_limits_FL_min
    return  out_of_limits #torch.sum(out_of_limits)

def feet_swip(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    hip_vel_pitch_l = asset.data.joint_vel[:, 4]
    hip_vel_pitch_r = asset.data.joint_vel[:, 5]

    reward = torch.abs(hip_vel_pitch_l-hip_vel_pitch_r)
    # print(reward)
    # reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward