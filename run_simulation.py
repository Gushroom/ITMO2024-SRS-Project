import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import logging
from estimators import groundtruth_estimator, RK4_estimator
from controllers import PIDController

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load the MuJoCo model
model_path = 'models/robot.xml' 
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
pos_x, pos_y, theta, v_linear, v_angular = [], [], [], [], []

# Initialize states for estimator
position = np.array([0.0, 0.0, 0.0])  # [x, y, z]
velocity = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz]
quat = np.array([1.0, 0.0, 0.0, 0.0])  # Orientation quaternion [w, x, y, z]


def get_state_from_simulation(data):
    x, y, quat = groundtruth_estimator(data)
    theta = get_yaw_from_quaternion(quat)
    return {"position": (x, y), "orientation": theta}

with mujoco.viewer.launch_passive(model, data) as viewer:
    controller_params = {
        "x_d": 3.0, "y_d": 3.0, 
        "K_p_pos": 2.5, "K_i_pos": 0.001, "K_d_pos": 0.5,
        "K_p_theta": 2.5, "K_i_theta": 0.001, "K_d_theta": 0.01,
        "wheelbase": 0.6, "V_MAX": 5.0
    }
    controller = PIDController(model, data, controller_params)
    start_time = time.time()
    last_log_time = -np.inf
    while viewer.is_running():
        dt = model.opt.timestep
        state = get_state_from_simulation(data)

        # Compute controls
        controls = controller.compute_controls(state, dt)
        
        # Apply controls to actuators
        data.ctrl[left_actuator_id] = controls["left_wheel"]
        data.ctrl[right_actuator_id] = controls["right_wheel"]

        current_time = time.time() - start_time
        # Log every second
        if current_time - last_log_time >= 1.0:
            logging.info(f"Time: {current_time:.2f}, {state}")
            last_log_time = current_time

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()

        # Sleep to maintain real-time simulation
        elapsed_time = time.time() - start_time
        if elapsed_time < data.time:
            time.sleep(data.time - elapsed_time)
