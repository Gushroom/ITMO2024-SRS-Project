import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import logging
from estimators import groundtruth_estimator, RK4_estimator
from controllers import PIDController, ScheduledPIDController, SlidingModeController
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
X_D = -5
Y_D = -3

with mujoco.viewer.launch_passive(model, data) as viewer:
    PID_controller_params = {
        "x_d": X_D, "y_d": Y_D, 
        "K_p_pos": 1.0, "K_i_pos": 0.0001, "K_d_pos": 0.3,
        "K_p_theta": 1.0, "K_i_theta": 0.0001, "K_d_theta": 0.2,
        "wheelbase": 0.6, "V_MAX": 5.0
    }
    controller = PIDController(model, data, PID_controller_params)

    # scheduled_PID_params = {
    #     "x_d": X_D, "y_d": Y_D, 
    #     "K_p_pos": 1.0, "K_i_pos": 0.0001, "K_d_pos": 0.3,
    #     "K_p_theta": 10.0, "K_i_theta": 0.0001, "K_d_theta": 0.2,
    #     "wheelbase": 0.6, "V_MAX": 5.0
    # }
    # controller = ScheduledPIDController(model, data, scheduled_PID_params)

    # SMC_controller_params = {
    #     "x_d": X_D, "y_d": Y_D,
    #     "c_pos": 2.0, "c_theta": 2.0,
    #     "k_pos": 5.0, "k_theta": 3.0,
    #     "epsilon": 0.01, "wheelbase": 0.6, "V_MAX": 5.0
    # }
    # controller = SlidingModeController(SMC_controller_params)   

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


        # Send a new control signal every interval
        if current_time - last_ctrl_time >= control_update_interval:
            # Compute controls
            controls = controller.compute_controls(state, control_update_interval)
            
            # Apply controls to actuators
            data.ctrl[left_actuator_id] = controls["left_wheel"]
            data.ctrl[right_actuator_id] = controls["right_wheel"]
            last_ctrl_time = current_time

        # Log every second
        if current_time - last_log_time >= 1.0:
            pos_x = state['position'][0]
            pos_y = state['position'][1]
            pos_t = math.degrees(state["orientation"])
            err_x = X_D - pos_x
            err_y = Y_D - pos_y
            theta_d = math.degrees(math.atan2(err_y, err_x))
            ctrl_l = controls["left_wheel"]
            ctrl_r = controls["right_wheel"]
            logging.info(f"Time: {current_time:.2f}, " \
                         f"Position: ({pos_x:.2f}, {pos_y:.2f}, {pos_t:.2f}), " \
                         f"Controls: Left: {ctrl_l:.2f}, Right: {ctrl_r:.2f}, " \
                         f"Desired Theta: {theta_d:.2f}")
            last_log_time = current_time

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()

        # Sleep to maintain real-time simulation
        elapsed_time = time.time() - start_time
        if elapsed_time < data.time:
            time.sleep(data.time - elapsed_time)