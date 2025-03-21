import torch
import time
import torch.nn as nn
import pandas as pd
import numpy as np
from Library.BRUCE_GYM.GAZEBO_INTERFACE import Manager as gazebo_manager
leg_p_gains = [265, 150,  80,  80,    30]
leg_i_gains = [  0,   0,   0,   0,     0]
leg_d_gains = [ 1., 2.3, 0.8, 0.8, 0.003]
arm_p_gains = [ 1.6,  1.6,  1.6]
arm_i_gains = [   0,    0,    0]
arm_d_gains = [0.03, 0.03, 0.03]
p_gains = leg_p_gains * 2 + arm_p_gains * 2  # the joint order matches the robot's sdf file
i_gains = leg_i_gains * 2 + arm_i_gains * 2
d_gains = leg_d_gains * 2 + arm_d_gains * 2
action_dof_map=[1,3,5,7,9,0,2,4,6,8]
obs_dof_map=[5,0,13,10,6,1,14,11,7,2,15,12,8,3,9,4]
last_ac_dof_map=[0,1,4,5,8,9,12,13,14,15]
class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Actor网络
        self.actor = nn.Sequential(
                    nn.Linear(60, 256),
                    nn.ELU(),
                    nn.Linear(256, 128),
                    nn.ELU(),
                    nn.Linear(128, 128),
                    nn.ELU(),
                    nn.Linear(128, 10)
                )
        
        # 添加缺失的Critic网络
        self.critic = nn.Sequential(
                    nn.Linear(60, 256),
                    nn.ELU(),
                    nn.Linear(256, 128),
                    nn.ELU(),
                    nn.Linear(128, 128),
                    nn.ELU(),
                    nn.Linear(128, 1)
                )
                
        # 添加缺失的标准差参数
        self.std = nn.Parameter(torch.ones(10))
    def forward(self, x):
            return self.actor(x)
class GazeboController:
    def __init__(self, robot_name="bruce", num_joints=16,last_action=[0]*16):
        # 初始化Gazebo接口
        self.gazebo_interface = gazebo_manager.GazeboInterface(robot_name, num_joints)
        self.gazebo_interface.set_operating_mode(2)#设置操作模式为position控制
        self.gazebo_interface.set_all_position_pid_gains(p_gains, i_gains, d_gains)#设置PID参数（copy from Startups/run_simulation.py）
        self.last_action=last_action

    def _get_observations(self):
        # Get IMU state
        imu_state = self.gazebo_interface.IMU_STATE.get()
        imu_ang_rate = imu_state['ang_rate']

        # Get body pose
        body_pose = self.gazebo_interface.BODY_POSE.get()
        body_quaternion = body_pose['quaternion']
        body_velocity = body_pose['velocity']
        # 根据四元数计算重力向量
        q=torch.tensor(body_quaternion,dtype=torch.float,device=torch.device('cpu')).unsqueeze(0)
        shape=q.shape
        v=torch.tensor([0,0,-1],dtype=torch.float,device=torch.device('cpu')).unsqueeze(0)    #v是重力之类的，默认为(0,0,-1)
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        projected_gravity= a - b + c
        projected_gravity = projected_gravity.numpy().flatten()
        velocity_command=[0.5,0,0]
        # Get joint positions, velocities, forces
        joint_state = self.gazebo_interface.JOINT_STATE.get()
        joint_positions = joint_state['position']
        joint_velocities = joint_state['velocity']
        #调整观察空间的角度顺序
        for i in range(16):
            joint_positions[i]=joint_positions[obs_dof_map[i]]
            joint_velocities[i]=joint_velocities[obs_dof_map[i]]
        #连接为60维的观测空间
        observation = np.concatenate([
            body_velocity,#3
            imu_ang_rate,#3
            projected_gravity,#3
            velocity_command,#3
            joint_positions,#16
            joint_velocities,#16
            self.last_action#16
        ])
        return observation
    def save_obs_to_excel(self, obs, file_path="observations.xlsx"):
        """
        将观测数据保存到Excel文件中
        """
        # 将观测数据转换为DataFrame
        columns = [
            "body_velocity_x", "body_velocity_y", "body_velocity_z",
            "imu_ang_rate_x", "imu_ang_rate_y", "imu_ang_rate_z",
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "velocity_command_x", "velocity_command_y", "velocity_command_z"
        ] + [f"joint_position_{i}" for i in range(16)] + \
                  [f"joint_velocity_{i}" for i in range(16)] + \
                  [f"last_action_{i}" for i in range(16)]

        df = pd.DataFrame([obs], columns=columns)

        # 如果文件已存在，则追加数据
        try:
            existing_data = pd.read_excel(file_path)
            df = pd.concat([existing_data, df], ignore_index=True)
        except FileNotFoundError:
            pass  # 文件不存在时直接创建新文件

        # 保存到Excel
        df.to_excel(file_path, index=False)

    def control_loop(self):
        # 初始化站立姿势(默认为0度)
        self.gazebo_interface.reset_simulation()
        self.model = Actor()
        map_location=torch.device('cpu')
        weights=torch.load("./model_99999.pt",map_location=map_location)
        #print(weights.keys())
        #go=input("keep going?")
        self.model.load_state_dict(weights['model_state_dict'])#加载模型权重参数
        self.model.eval()#启动评估模式
        with torch.no_grad():
            test_input = torch.ones(60, dtype=torch.float32).unsqueeze(0)  # 生成全为1.0的测试输入
            output = self.model(test_input)  # 添加 batch 维度
            print("模型输出:", output)
            go=input("keep going?")

        # 主控制循环
        while True:
            try:
                # 获取观测数据
                obs = self._get_observations()
                self.save_obs_to_excel(obs=obs)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                print("Observation shape:", obs_tensor.shape)
                
                # 生成控制指令
                action = self.model(obs_tensor)
                # 将动作转换为numpy数组
                action = action.squeeze(0).detach().numpy()
                # 将动作转换为列表
                action = action.astype(np.float32).tolist()
                print(f"Action:\n{action}")
                for i in range(10):
                    self.last_action[last_ac_dof_map[i]]=action[i]#更新上一次动作
                # 发送关节指令（按映射关系排序）
                final_action=[0]*16
                for i in range(10):
                    final_action[i]=action[action_dof_map[i]]#调整腿部关节角度顺序
                self.gazebo_interface.set_command_position(final_action)
                
                # 步进仿真
                self.gazebo_interface.step_simulation()
                body_pose = self.gazebo_interface.BODY_POSE.get()
                body_position = body_pose['position']
                if body_position[2] < 0.25:
                    go=input("keep going?(press n to stop!)")
                    if go=='n':
                        break
                    self.gazebo_interface.reset_simulation()
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    controller = GazeboController()
    controller.control_loop()