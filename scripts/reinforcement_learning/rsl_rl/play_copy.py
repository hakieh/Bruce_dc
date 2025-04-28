# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.task ==None:
   args_cli.task = "Isaac-Velocity-Flat-Bruce-v0"
if args_cli.num_envs == None:
    args_cli.num_envs = 4
# args_cli.checkpoint="2025-03-24_19-18-17/model_1000.pt."
#a
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.load_checkpoint:
        agent_cfg.load_checkpoint = args_cli.load_checkpoint
    # specify directory for logging experiments
    log_root_path = os.path.join("/home/kh/hkh/data/weight/logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        # resume_path = retrieve_file_path(os.path.join(log_root_path,args_cli.load))
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        print(agent_cfg.load_checkpoint,agent_cfg.load_run)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)#
    # print("loding model from: ",resume_path)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path,load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
#     obs = torch.tensor([0.0976270078546495 , 0.43037873274483895 , 0.20552675214328775 , 0.08976636599379373 , -0.15269040132219058 , 0.29178822613331223 , -0.12482557747461498 , 0.7835460015641595 , 0.9273255210020586 , -0.2331169623484446 ,
# 0.5834500761653292 , 0.05778983950580896 , 0.13608912218786462 , 0.8511932765853221 , -0.8579278836042261 , -0.8257414005969186 , -0.9595632051193486 , 0.665239691095876 , 0.556313501899701 , 0.7400242964936383 ,
# 0.957236684465528 , 0.5983171284334472 , -0.07704127549413631 , 0.5610583525729109 , -0.7634511482621336 , 0.27984204265504764 , -0.7132934251819072 , 0.8893378340991678 , 0.04369664350014335 , -0.17067612001895283 ,
# -0.47088877579074606 , 0.5484673788684333 , -0.0876993355669029 , 0.13686789773729702 , -0.9624203991272897 , 0.23527099415175412 , 0.22419144544484282 , 0.23386799374951384 , 0.8874961570292483 , 0.3636405982069668 ,
# -0.280984198852428 , -0.1259360924013171 , 0.3952623918545297 , -0.8795490567414603 , 0.33353343089133536 , 0.3412757392363188 , -0.5792348778523182 , -0.7421474046902934 , -0.36914329815163227 , -0.2725784581147548 ,
# 0.14039354083575928 , -0.1227969730753593 , 0.9767476761184524 , -0.7959103785039439 , -0.5822464878103306 , -0.6773809642300075 , 0.30621665093079686 , -0.4934167949204358 , -0.06737845428738742 , -0.5111488159967945 ],device="cuda:0")
#     actions = policy(obs)
#     print(actions)
#     return
    timestep = 0
    time_log = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # simulate environment
    actions_history = []
    # actions_history = []
    actions_his_np = []
    obs_hisotry = []
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # obs = torch.ones_like(obs)
            actions = policy(obs)
            # actions_history.append(actions.squeeze(0))
            actions_history.append(actions.squeeze(0)[0,:])
            actions_his_np.append(actions.squeeze(0).cpu().numpy()[0,:])
            obs_hisotry.append(obs.squeeze(0).cpu().numpy()[0,:])
            # print(actions,"actions")
            # print("==========================")
            # return
            # env stepping
            obs, _, _, _ = env.step(actions)
            if len(actions_history) == 500:
                print(len(actions_history))
                result_tensor = torch.stack(actions_history, dim=0)
                result_tensor = result_tensor.cpu().numpy()
                # if args_cli.num_envs ==1:
                obs_hisotry_pd = pd.DataFrame(obs_hisotry)
                actions_history_pd = pd.DataFrame(actions_his_np)
                np.save(f"/home/kh/hkh/data/excle/4.27/afop_bruce_actions_history_{time_log}.npy",result_tensor)
                print("save actions history!")
                path_actions = "/home/kh/hkh/data/excle/4.27/afop_bruce_actions"+time_log+".xlsx"
                with pd.ExcelWriter(path_actions,'auto') as xlsx:
                    actions_history_pd.to_excel(xlsx)
                path_obs = "/home/kh/hkh/data/excle/4.27/afop_bruce_obs"+time_log+".xlsx"
                with pd.ExcelWriter(path_obs,'auto') as xlsx:
                    obs_hisotry_pd.to_excel(xlsx)
            # if len(actions_history) == 500:
            #     result_tensor = torch.stack(actions_history, dim=0)
            #     result_tensor = result_tensor.cpu().numpy()
            #     np.save(f"/home/kh/hkh/code/isaaclab_pip_1/weight/actions_history_{time_log}.npy",result_tensor)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
