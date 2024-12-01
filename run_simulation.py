import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import logging
from estimators import groundtruth_estimator, IMU_vel_estimator, motion_model
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
left_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_wheel')
right_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_wheel')


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


def get_state_from_simulation(data):
    x, y, quat = groundtruth_estimator(data)
    theta = get_yaw_from_quaternion(quat)
    return {"position": (x, y), "orientation": theta}


# Desired position
# target_positions = [(0,0.01), (0, 0.02)]
target_positions = []
r = 2
theta = 0
step = 0.01

while theta < 2 * math.pi:
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    target_positions.append((x, y))
    theta += step
current_target = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    vel_controller_params = {
        "K_p_pos": 1.0, "K_i_pos": 0.001, "K_d_pos": 0.3,
        "K_p_theta": 1.0, "K_i_theta": 0.001, "K_d_theta": 0.3,
        "V_MAX": 5.0, "wheelbase": 0.6
    }
    vel_controller = VelocityPID(vel_controller_params)

    # vel_controller_params = {
    #     "K_pos": 1.5, "K_theta": 1.5,
    #     "lambda_pos": 0.5, "lambda_theta": 0.5,
    #     "epsilon_pos": 1.0, "epsilon_theta": 0.3,
    #     "V_MAX": 5.0, "wheelbase": 0.6
    #     }
    # vel_controller = VelocitySlidingMode(vel_controller_params)

    acc_controller_params = {
        "K_p": 5.0, "K_i": 0.001, "K_d": 0.1,
        "wheelradius": 0.2, "T_MAX": 10, "T_MIN": -10
    }
    acc_controller = AccelerationPID(acc_controller_params)
    start_time = time.time()
    last_log_time = -np.inf
    last_ctrl_time = -np.inf
    control_update_interval = 0.1
    prev_state = {"position": (0.0, 0.0), "orientation": 0.0}
    err_x = 0
    err_y = 0
    theta_d = 0
    ctrl_l = 0.0
    ctrl_r = 0.0
    prev_v = 0.0
    while viewer.is_running():
        current_time = time.time() - start_time
        dt = model.opt.timestep

        if current_target < len(target_positions):
            X_D = target_positions[current_target][0]
            Y_D = target_positions[current_target][1]

            if current_time - last_ctrl_time >= control_update_interval:
                accel_data = data.sensordata[imu_acc_sensor_adr:imu_acc_sensor_adr + 3]
                gyro_data = data.sensordata[imu_gyro_sensor_adr:imu_gyro_sensor_adr + 3]

                v, w = IMU_vel_estimator(acc=accel_data, gyro=gyro_data, prev_v=prev_v, dt=control_update_interval)
                prev_v = v

                state = motion_model(v, w, prev_state, control_update_interval)
                prev_state = state

                x, y = state["position"]
                theta = state["orientation"]

                err_x = X_D - x
                err_y = Y_D - y

                velocity_controls = vel_controller.compute_controls(state, X_D, Y_D, control_update_interval)
                desired_velocity = (velocity_controls["left_wheel"], velocity_controls["right_wheel"])

                qvel_start_index = model.body_dofadr[body_id]
                # [vx, vy, vz]
                linear_velocity_vec = data.qvel[qvel_start_index : qvel_start_index + 3]
                v_x, v_y, v_z = linear_velocity_vec
                # [v, w]
                linear_velocity = v_x * np.cos(theta) + v_y * np.sin(theta)
                angular_velocity = data.qvel[qvel_start_index + 3:qvel_start_index + 6][2]

                # Get the indices into data.qvel for the wheel joints
                omega_left_idx = model.jnt_dofadr[left_wheel_id]
                omega_right_idx = model.jnt_dofadr[right_wheel_id]
                omega_left = data.qvel[omega_left_idx]
                omega_right = data.qvel[omega_right_idx]

                # print(f"Actual Linear: {linear_velocity} \nEstimated linear: {v_est}")

                # actual_velocity = (linear_velocity, angular_velocity)
                actual_velocity = (omega_left, omega_right)
                torque_controls = acc_controller.compute_controls(desired_velocity, actual_velocity, control_update_interval)

                data.ctrl[left_actuator_id] = torque_controls["tau_left"]
                data.ctrl[right_actuator_id] = torque_controls["tau_right"]

                # state_estimation = get_state_from_kalman(torque_controls["tau_right"],
                #                          torque_controls["tau_left"],
                #                          gyro_data, accel_data)
                last_ctrl_time = current_time

            # Log every second
            if current_time - last_log_time >= 1.0:
                logging.info(f"Time: {current_time:.2f}, " \
                            f"Position: ({x:.2f}, {y:.2f}, {theta:.2f}), " \
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