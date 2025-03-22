尝试部署：
#请确保Simulation/config.py中，fixed=False（使BRUCE可以落在地面上）
1.启动gazebo仿真
#进入目录
cd Bruce-op（按照自己电脑上的文件地址来）
#启动gazebo
Simulation/gazebo_simulation.launch
2.新开终端，运行文件即可
python3 load_kaihui_model.py
#运行后会得到observations.xlsx，里面记录了每次返回的observation

仅测试关节运动：
==========仿真==========
#请确保Simulation/config.py中，fixed=True（将BRUCE固定在空中）
1.启动gazebo仿真
#进入目录
cd Bruce-op（按照自己电脑上的文件地址来）
#启动gazebo
Simulation/gazebo_simulation.launch
2.创建共享内存
python3 -m Startups.memory_manager
3.初始化BRUCE（初始化相关数据）
python3 -m Startups.run_simulation
运行预设动作（可根据需要更改）
python3 set_joint_position.py
==========实机==========
#进入目录
cd Bruce-op（按照自己电脑上的文件地址来）
终端1：
python3 -m Startups.memory_manager
Startups/usb_latency_setup.sh
python3 -m Startups.run_bear（启动腿部电机）
终端2：
python3 -m Startups.run_dxl（启动手臂舵机）
#注意：执行此动作后，BRUCE手臂将伸直，因此在执行前，请务必确认手臂两旁无遮挡！！！
运行预设动作（可根据需要更改）
终端3：
python3 set_joint_position.py