import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import logging
from estimators import groundtruth_estimator, RK4_estimator
from controllers import VelocityPID, AccelerationPID, VelocitySlidingMode
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load the MuJoCo model
# model_path = 'models/robot.xml' 
model_path = 'test_robot.xml' 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Get actuators and sensors
left_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_velocity_servo")
right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_velocity_servo")

imu_gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
imu_acc_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accelerometer")
imu_gyro_sensor_adr = model.sensor_adr[imu_gyro_sensor_id]
imu_acc_sensor_adr = model.sensor_adr[imu_acc_sensor_id]

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_body')


# Initialize control inputs
controls = np.zeros(model.nu)  # Number of actuators

# Function to extract yaw (heading angle) from quaternion
def get_yaw_from_quaternion(quat):
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

# Initialize for visualization
traj_x, traj_y, traj_theta, v_linear, v_angular = [], [], [], [], []

# Initialize states for estimator
position = np.array([0.0, 0.0, 0.0])  # [x, y, z]
velocity = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz]
quat = np.array([1.0, 0.0, 0.0, 0.0])  # Orientation quaternion [w, x, y, z]


def get_state_from_simulation(data):
    x, y, quat = groundtruth_estimator(data)
    theta = get_yaw_from_quaternion(quat)
    return {"position": (x, y), "orientation": theta}

# Desired position
target_positions = [(3, 3), (3, -3), (-3, -3), (-3, 3), (0, 0)]
current_target = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    vel_controller_params = {
        "K_p_pos": 1.0, "K_i_pos": 0.001, "K_d_pos": 0.5,
        "K_p_theta": 1.0, "K_i_theta": 0.001, "K_d_theta": 0.5,
        "V_MAX": 5.0, "wheelbase": 0.6
    }
    vel_controller = VelocityPID(vel_controller_params)

    # vel_controller_params = {
    #     "K_pos": 1, "K_theta": 3,
    #     "lambda_pos": 0.03, "lambda_theta": 0.03,
    #     "epsilon_pos": 1, "epsilon_theta": 1,
    #     "V_MAX": 5.0, "wheelbase": 0.6
    #     }
    # vel_controller = VelocitySlidingMode(vel_controller_params)

    acc_controller_params = {
        "K_p": 2.0, "K_i": 0.001, "K_d": 0.3,
        "wheelbase": 0.6, "T_MAX": 10, "T_MIN": -10
    }
    acc_controller = AccelerationPID(acc_controller_params)
    start_time = time.time()
    last_log_time = -np.inf
    last_ctrl_time = -np.inf
    control_update_interval = 0.1
    while viewer.is_running():
        current_time = time.time() - start_time
        dt = model.opt.timestep
        state = get_state_from_simulation(data)
        traj_x.append(state["position"][0])
        traj_y.append(state["position"][1])
        traj_theta.append(state["orientation"])

        if current_target < len(target_positions):
            X_D = target_positions[current_target][0]
            Y_D = target_positions[current_target][1]

            pos_x = state['position'][0]
            pos_y = state['position'][1]
            pos_t = math.degrees(state["orientation"])
            err_x = X_D - pos_x
            err_y = Y_D - pos_y
            theta_d = math.degrees(math.atan2(err_y, err_x))
            ctrl_l = 0.0
            ctrl_r = 0.0
            # Send a new control signal every interval
            if current_time - last_ctrl_time >= control_update_interval:
                # Compute controls
                # Unpack velocity control commands
                velocity_controls = vel_controller.compute_controls(state, X_D, Y_D, control_update_interval)
                desired_velocity = (velocity_controls["left_wheel"], velocity_controls["right_wheel"])
                # Actural v and w from state
                qvel_start_index = model.body_dofadr[body_id]
                linear_velocity = data.qvel[qvel_start_index : qvel_start_index + 3]
                angular_velocity = data.qvel[qvel_start_index + 3:qvel_start_index +  6]
                actural_velocity = (linear_velocity, angular_velocity)
                torque_controls = acc_controller.compute_controls(desired_velocity, actural_velocity, control_update_interval)
                # Apply controls to actuators
                # data.ctrl[left_actuator_id] = controls["left_wheel"]
                # data.ctrl[right_actuator_id] = controls["right_wheel"]
                data.ctrl[left_actuator_id] = torque_controls["tau_left"]
                data.ctrl[right_actuator_id] = torque_controls["tau_right"]
                last_ctrl_time = current_time

            # Log every second
            if current_time - last_log_time >= 1.0:
                logging.info(f"Time: {current_time:.2f}, " \
                            f"Position: ({pos_x:.2f}, {pos_y:.2f}, {pos_t:.2f}), " \
                            f"Velocity Commands: ({desired_velocity[0]:.2f}, {desired_velocity[1]:.2f}), "
                            f"Controls: Left: {torque_controls["tau_left"]:.2f}, Right: {torque_controls["tau_right"]:.2f}, " \
                            f"Desired Theta: {theta_d:.2f}. Target Index: {current_target}")
                last_log_time = current_time

            if np.hypot(err_x, err_y) < 0.25:
                logging.info("Moving on to next target.")
                current_target += 1

        if current_target >= len(target_positions):
            logging.info("Tracing complete.")
            break

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()

        # Sleep to maintain real-time simulation
        elapsed_time = time.time() - start_time
        if elapsed_time < data.time:
            time.sleep(data.time - elapsed_time)