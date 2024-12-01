import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import logging
from estimators import groundtruth_estimator, RK4_estimator, KalmanFilter
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


kf = KalmanFilter(r=0.2, L=0.6, I_wheel=0.04, I_body=0.4167, m_robot=16.0)
def get_state_from_kalman(tau_r, tau_l, gyro, acc):
    dt = 0.1

    # Predict
    kf.predict(tau_r, tau_l, dt)

    # Update
    kf.update(gyro, acc, dt)

    return kf.get_state()

rk4 = RK4_estimator(radius=0.2, wheelbase=0.6)
def get_state_from_rk4(omega_left, omega_right, dt):
    return rk4.estimate(omega_left, omega_right, dt)


# Desired position
# target_positions = [(3, 3), (3, -3), (-3, -3), (-3, 3), (0, 0)]
target_positions = [(-3, -3), (0, 0)]
current_target = 0

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

# Initialize variables
start_time = time.time()
last_ctrl_time = -np.inf
last_log_time = -np.inf
control_update_interval = 0.1
current_target = 0
traj_x = []
traj_y = []
est_x = []
est_y = []
traj_theta = []
v_actural = []
w_actural = []
v_estimation = []
w_estimation = []

# Main simulation loop
while True:
    current_time = time.time() - start_time
    dt = model.opt.timestep
    state = get_state_from_simulation(data)
    x, y = state["position"]
    theta = state["orientation"]
    traj_x.append(x)
    traj_y.append(y)
    traj_theta.append(theta)

    if current_target < len(target_positions):
        X_D = target_positions[current_target][0]
        Y_D = target_positions[current_target][1]

        pos_x = x
        pos_y = y
        pos_t = math.degrees(theta)
        err_x = X_D - pos_x
        err_y = Y_D - pos_y
        theta_d = math.degrees(math.atan2(err_y, err_x))
        ctrl_l = 0.0
        ctrl_r = 0.0

        if current_time - last_ctrl_time >= control_update_interval:
            velocity_controls = vel_controller.compute_controls(state, X_D, Y_D, control_update_interval)
            desired_velocity = (velocity_controls["left_wheel"], velocity_controls["right_wheel"])

            qvel_start_index = model.body_dofadr[body_id]
            # [vx, vy, vz]
            linear_velocity_vec = data.qvel[qvel_start_index : qvel_start_index + 3]
            v_x, v_y, v_z = linear_velocity_vec
            # [v, w]
            linear_velocity = v_x * np.cos(theta) + v_y * np.sin(theta)
            angular_velocity = data.qvel[qvel_start_index + 3:qvel_start_index + 6][2]
            accel_data = data.sensordata[imu_acc_sensor_adr:imu_acc_sensor_adr + 3]
            gyro_data = data.sensordata[imu_gyro_sensor_adr:imu_gyro_sensor_adr + 3]
            v_actural.append(linear_velocity)
            w_actural.append(angular_velocity)
            # Get the indices into data.qvel for the wheel joints
            omega_left_idx = model.jnt_dofadr[left_wheel_id]
            omega_right_idx = model.jnt_dofadr[right_wheel_id]
            omega_left = data.qvel[omega_left_idx]
            omega_right = data.qvel[omega_right_idx]
            print(f"Omega left: {omega_left} Omega right: {omega_right}")
            # Compute linear and angular velocities from wheel speeds
            v_est = (0.2 / 2) * (omega_left + omega_right)
            omega_est = (0.2 / 0.6) * (omega_right - omega_left)
            v_estimation.append(v_est)
            w_estimation.append(omega_est)

            # Adjust the estimated linear velocity to account for robot's heading
            # Integrate over time to update robot's orientation (theta)
            delta_time = control_update_interval  # Time step (adjust according to your simulation's time step)
            theta += omega_est * delta_time  # Update robot's heading

            # Project the linear velocity onto the robot's heading
            v_est_x = v_est * np.cos(theta)
            v_est_y = v_est * np.sin(theta)

            print(f"Actual Linear: {linear_velocity} \nEstimated linear: {v_est}")

            # actural_velocity = (linear_velocity, angular_velocity)
            actural_velocity = (omega_left, omega_right)
            torque_controls = acc_controller.compute_controls(desired_velocity, actural_velocity, control_update_interval)

            data.ctrl[left_actuator_id] = torque_controls["tau_left"]
            data.ctrl[right_actuator_id] = torque_controls["tau_right"]

            # state_estimation = get_state_from_kalman(torque_controls["tau_right"],
            #                          torque_controls["tau_left"],
            #                          gyro_data, accel_data)
            state_estimation = get_state_from_rk4(omega_left, omega_right, control_update_interval)
            est_x.append(state_estimation[0])
            est_y.append(state_estimation[1])
            last_ctrl_time = current_time

        if current_time - last_log_time >= 1.0:
            logging.info(f"Time: {current_time:.2f}, " \
                        f"Position: ({pos_x:.2f}, {pos_y:.2f}, {pos_t:.2f}), " \
                        f"Velocity Commands: ({desired_velocity[0]:.2f}, {desired_velocity[1]:.2f}), " \
                        f"Controls: Left: {torque_controls['tau_left']:.2f}, Right: {torque_controls['tau_right']:.2f}, " \
                        f"Desired Theta: {theta_d:.2f}. Target Index: {current_target}")
            last_log_time = current_time

        if np.hypot(err_x, err_y) < 0.25:
            logging.info("Moving on to next target.")
            current_target += 1

    else:
        logging.info("Tracing complete.")
        break

    # Step simulation
    mujoco.mj_step(model, data)

    # Optional: Sleep to maintain real-time simulation
    elapsed_time = time.time() - start_time
    if elapsed_time < data.time:
        time.sleep(data.time - elapsed_time)


plt.figure()
plt.plot(traj_x, traj_y, label='Trajectory')
plt.scatter([pos[0] for pos in target_positions], [pos[1] for pos in target_positions], color='red', label='Targets')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Vehicle Trajectory')
plt.legend()
plt.show()


plt.figure()
plt.plot(est_x, est_y, label='Trajectory')
plt.scatter([pos[0] for pos in target_positions], [pos[1] for pos in target_positions], color='red', label='Targets')
plt.xlabel('Estimated X position')
plt.ylabel('Estimated Y position')
plt.title('Estimated Vehicle Trajectory')
plt.legend()
plt.show()


plt.figure()
plt.plot(v_actural, label='v_actual')
plt.plot(v_estimation, label='v_estimation')
plt.title('Linear Velocity Comparison')
plt.xlabel('Timestep')
plt.ylabel('Linear Velocity (m/s)')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(w_actural, label='w_actual')
plt.plot(w_estimation, label='w_estimation')
plt.title('Angular Velocity Comparison')
plt.xlabel('Timestep')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid()
plt.show()

