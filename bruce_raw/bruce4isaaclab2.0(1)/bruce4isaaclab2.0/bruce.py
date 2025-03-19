import isaaclab.sim  as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG



BRUCE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/data/hkh/code/IsaacLab_bipedal/source/extensions/omni.isaac.lab_assets/BRUCE_simulation_models-1.6/usd/bruce/bruce.usd",
        usd_path="C:/PythonProject/bruce.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            # ".*_hip_pitch_joint": -0.20,
            # ".*_knee_joint": 0.42,
            # ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_pitch_joint": 0.87,
            # "left_shoulder_roll_joint": 0.16,
            # "left_shoulder_pitch_joint": 0.35,
            # "right_shoulder_roll_joint": -0.16,
            # "right_shoulder_pitch_joint": 0.35,
            # "left_one_joint": 1.0,
            # "right_one_joint": -1.0,
            # "left_two_joint": 0.52,
            # "right_two_joint": -0.52,
            'hip_yaw_l' :0.,   # [rad]
            'hip_roll_l':0.,   # [rad]
            'hip_pitch_l' :0.,   # [rad]
            'knee_pitch_l' :0.,   # [rad]
            'ankle_pitch_l':0.,   # [rad]
            # 'hip_yaw_r' :2.,   # [rad]
            'hip_yaw_r' :1,   # [rad]
            'hip_roll_r' :0.,   # [rad]
            # 'hip_pitch_r' :0.,   # [rad]
            'hip_pitch_r' :1,   # [rad]
            'knee_pitch_r' :0.,   # [rad]
            'ankle_pitch_r' :0.,   # [rad]
            # 'shoulder_pitch_l' :2.35,   # [rad]
            'shoulder_pitch_l' :0.,   # [rad]
            # 'shoulder_roll_l' :-0.16,   # [rad]
            'shoulder_roll_l' :0.,   # [rad]
            # 'elbow_pitch_l' :0.,   # [rad]
            'elbow_pitch_l' :1,   # [rad]
            # 'shoulder_pitch_r' :0.35,   # [rad]
            'shoulder_pitch_r' :0.,   # [rad]
            # 'shoulder_roll_r' :-0.16,   # [rad]
            'shoulder_roll_r' :0.,   # [rad]
            'elbow_pitch_r' :0.,   # [rad]
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,


    # actuators={
    #     "legs": DCMotorCfg(
    #         saturation_effort=23.7,
    #         joint_names_expr=[
    #             "hip_yaw_l",
    #             "hip_roll_l",
    #             "hip_pitch_l",
    #             "knee_pitch_l",
    #             "hip_yaw_r",
    #             "hip_roll_r",
    #             "hip_pitch_r",
    #             "knee_pitch_r",
                
    #         ],
    #         effort_limit=300,
    #         velocity_limit=100.0,
    #         stiffness={
    #             "hip_yaw_r": 25,
    #             "hip_roll_r":  25,
    #             "hip_pitch_r":  25,
    #             "knee_pitch_r":  25,
    #             "hip_yaw_l":  25,
    #             "hip_roll_l": 25,
    #             "hip_pitch_l":  25,
    #             "knee_pitch_l":  25,
    #             # "torso_joint": 200.0,
    #         },
    #         damping={
    #             # ".*_hip_yaw_joint": 5.0,
    #             # ".*_hip_roll_joint": 5.0,
    #             # ".*_hip_pitch_joint": 5.0,
    #             # ".*_knee_joint": 5.0,
    #             # "torso_joint": 5.0,
    #             "hip_yaw_r": 0.5,
    #             "hip_roll_r": 0.5,
    #             "hip_pitch_r":  0.5,
    #             "knee_pitch_r": 0.5,
    #             "hip_yaw_l":  0.5,
    #             "hip_roll_l":  0.5,
    #             "hip_pitch_l":  0.5,
    #             "knee_pitch_l":  0.5,
    #         },
    #         armature={
    #             "hip_yaw_r": 0.01,
    #             "hip_roll_r": 0.01,
    #             "hip_pitch_r": 0.01,
    #             "knee_pitch_r": 0.01,
    #             "hip_yaw_l": 0.01,
    #             "hip_roll_l": 0.01,
    #             "hip_pitch_l": 0.01,
    #             "knee_pitch_l": 0.01,
    #         },
    #     ),
    #     "feet": DCMotorCfg(
    #         saturation_effort=23.7,
    #         effort_limit=20,
    #         joint_names_expr=["ankle_pitch_r","ankle_pitch_l"],
    #         stiffness=25.0,
    #         damping=0.5,
    #         armature=0.01,
    #     ),
    #     "arms":  DCMotorCfg(
    #         saturation_effort=23.7,
    #         joint_names_expr=[
    #             "shoulder_pitch_r",
    #             "shoulder_roll_r",
    #             "elbow_pitch_r",
    #             "shoulder_pitch_l",
    #             "shoulder_roll_l",
    #             "elbow_pitch_l",
    #         ],
    #         effort_limit=300,
    #         velocity_limit=100.0,
    #         stiffness= 25,
    #         damping=0.5,
    #         armature={
    #             "shoulder_pitch_r":0.01,
    #             "shoulder_roll_r":0.01,
    #             "elbow_pitch_r":0.01,
    #             "shoulder_pitch_l":0.01,
    #             "shoulder_roll_l":0.01,
    #             "elbow_pitch_l":0.01,
    #             # ".*_five_joint": 0.001,
    #             # ".*_three_joint": 0.001,
    #             # ".*_six_joint": 0.001,
    #             # ".*_four_joint": 0.001,
    #             # ".*_zero_joint": 0.001,
    #             # ".*_one_joint": 0.001,
    #             # ".*_two_joint": 0.001,
    #         },
    #     ),
    # },


    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_yaw_l",
                "hip_roll_l",
                "hip_pitch_l",
                "knee_pitch_l",
                "hip_yaw_r",
                "hip_roll_r",
                "hip_pitch_r",
                "knee_pitch_r",
                
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "hip_yaw_r": 50,
                "hip_roll_r":  50,
                "hip_pitch_r":  50,
                "knee_pitch_r":  150,
                "hip_yaw_l":  150,
                "hip_roll_l": 150,
                "hip_pitch_l":  150,
                "knee_pitch_l":  150,
                # "torso_joint": 200.0,
            },
            damping={
                # ".*_hip_yaw_joint": 5.0,
                # ".*_hip_roll_joint": 5.0,
                # ".*_hip_pitch_joint": 5.0,
                # ".*_knee_joint": 5.0,
                # "torso_joint": 5.0,
                "hip_yaw_r": 5,
                "hip_roll_r": 5,
                "hip_pitch_r":  5,
                "knee_pitch_r": 5,
                "hip_yaw_l":  5,
                "hip_roll_l":  5,
                "hip_pitch_l":  5,
                "knee_pitch_l":  5,
            },
            armature={
                "hip_yaw_r": 0.01,
                "hip_roll_r": 0.01,
                "hip_pitch_r": 0.01,
                "knee_pitch_r": 0.01,
                "hip_yaw_l": 0.01,
                "hip_roll_l": 0.01,
                "hip_pitch_l": 0.01,
                "knee_pitch_l": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=["ankle_pitch_r","ankle_pitch_l"],
            stiffness=50.0,
            damping=5,
            armature=0.01,
        ),
        "arms":  ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_r",
                "shoulder_roll_r",
                "elbow_pitch_r",
                "shoulder_pitch_l",
                "shoulder_roll_l",
                "elbow_pitch_l",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness= 25,
            damping=1,
            armature={
                "shoulder_pitch_r":0.01,
                "shoulder_roll_r":0.01,
                "elbow_pitch_r":0.01,
                "shoulder_pitch_l":0.01,
                "shoulder_roll_l":0.01,
                "elbow_pitch_l":0.01,
                # ".*_five_joint": 0.001,
                # ".*_three_joint": 0.001,
                # ".*_six_joint": 0.001,
                # ".*_four_joint": 0.001,
                # ".*_zero_joint": 0.001,
                # ".*_one_joint": 0.001,
                # ".*_two_joint": 0.001,
            },
        ),
    },
)
