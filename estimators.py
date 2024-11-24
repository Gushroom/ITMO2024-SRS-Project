import numpy as np

# Estimator implementations
def groundtruth_estimator(data):
    '''
    This estimator uses ground truth position provided by the simulation environment
    '''
    x = data.qpos[0]  # Current x position
    y = data.qpos[1]  # Current y position
    quat = data.qpos[3:7]  # Orientation quaternion

    return x, y, quat

def RK4_estimator(gyro, acc, dt, position, velocity, quat):
    """
    RK4 estimator to integrate IMU data and estimate position and orientation.
    
    Parameters:
    - gyro: Angular velocity readings from gyroscope [wx, wy, wz]
    - acc: Linear acceleration readings from accelerometer [ax, ay, az]
    - dt: Time step for integration
    - position: Current position [x, y, z]
    - velocity: Current velocity [vx, vy, vz]
    - quat: Current orientation quaternion [w, x, y, z]
    
    Returns:
    - Updated position [x, y, z]
    - Updated velocity [vx, vy, vz]
    - Updated orientation quaternion [w, x, y, z]
    """

    # Functions for RK4 integration
    def derivative_state(state, input_gyro, input_acc):
        """
        Compute derivatives of position, velocity, and orientation.
        """
        pos = state[:3]    # Extract position
        vel = state[3:6]   # Extract velocity
        quat = state[6:]   # Extract quaternion
        
        # Angular velocity to quaternion derivative
        omega = np.array([0.0, input_gyro[0], input_gyro[1], input_gyro[2]])
        quat_dot = 0.5 * quaternion_multiply(quat, omega)
        
        # Acceleration
        acc_world = rotate_vector(input_acc, quat)  # Rotate acceleration to world frame
        acc_world[2] -= 9.81  # Subtract gravity
        
        return np.concatenate((vel, acc_world, quat_dot))
    
    def integrate_rk4(state, gyro, acc, dt):
        """
        RK4 integration for state update.
        """
        k1 = derivative_state(state, gyro, acc)
        k2 = derivative_state(state + 0.5 * dt * k1, gyro, acc)
        k3 = derivative_state(state + 0.5 * dt * k2, gyro, acc)
        k4 = derivative_state(state + dt * k3, gyro, acc)
        
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Concatenate state [position, velocity, quaternion]
    state = np.concatenate((position, velocity, quat))
    
    # Integrate using RK4
    state = integrate_rk4(state, gyro, acc, dt)
    
    # Extract updated values
    position = state[:3]
    velocity = state[3:6]
    quat = state[6:]
    
    return position, velocity, quat


def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def rotate_vector(vector, quat):
    """
    Rotate a vector using a quaternion.
    """
    quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    vector_quat = np.array([0.0] + vector.tolist())
    rotated = quaternion_multiply(quaternion_multiply(quat, vector_quat), quat_conj)
    return rotated[1:]  # Return the vector part
