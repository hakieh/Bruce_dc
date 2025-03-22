尝试部署：
#请确保Simulation/config.py中，fixed=False（使BRUCE可以落在地面上）
1.启动gazebo仿真
#进入目录
cd Bruce-op（按照自己电脑上的文件地址来）
#启动仿真
Simulation/gazebo_simulation.launch
2.新开终端，运行文件即可
python3 load_kaihui_model.py

仅测试关节运动：
#请确保Simulation/config.py中，fixed=True（将BRUCE固定在空中）
1.启动gazebo仿真
#进入目录
cd Bruce-op（按照自己电脑上的文件地址来）
#启动仿真
Simulation/gazebo_simulation.launch
2.创建共享内存
python3 -m Startups.memory_manager
3.启动仿真（BRUCE将初始化动作）
python3 -m Startups.run_simulation
4.运行预设动作
python3 set_joint_position.py