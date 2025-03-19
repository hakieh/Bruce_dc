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
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
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
    reward = torch.sum(torch.abs(ankle_angle), dim=1)
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



    out_of_limits_FR = -(asset.data.joint_pos[:, 12] - 1).clip(min=0.0)
    out_of_limits_FL = -(asset.data.joint_pos[:, 13] - 1).clip(min=0.0)

    out_of_limits_FR_min = (asset.data.joint_pos[:, 12] + 1).clip(max=0.0)
    out_of_limits_FL_min = (asset.data.joint_pos[:, 13] + 1).clip(max=0.0)

    # calf_limits_FR_min = (-1.5 - asset.data.joint_pos[:, 8] ).clip(max=0.0)
    # calf_limits_FL_min = (-1.5 - asset.data.joint_pos[:, 9] ).clip(max=0.0)

    # print(asset.data.joint_pos[:, [6,7]])
    # ooo

    
    # print(fl,rl,fr,rr,"--------")
    # print(asset.data.soft_joint_pos_limits,"=======================")
    # print(asset.data.joint_names,"=======================") ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

    # print(asset_cfg.joint_ids)
    # out_of_limits =  #fl + rl + fr + rr + out_of_limits_RR+ out_of_limits_RL + out_of_limits_RR_min + out_of_limits_RL_min + calf_limits_RR_min + calf_limits_RL_min\
    out_of_limits = out_of_limits_FR + out_of_limits_FL + out_of_limits_FR_min + out_of_limits_FL_min  #+ calf_limits_FR_min + calf_limits_FL_min
    return  out_of_limits #torch.sum(out_of_limits)

