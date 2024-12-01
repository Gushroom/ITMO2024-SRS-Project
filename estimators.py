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

class RK4_estimator():
    def __init__(self, wheelbase, radius, init_state=[0.0, 0.0, 0.0]):
        self.state = np.array(init_state)
        self.L = wheelbase
        self.r = radius

    def state_function(self, state, omega_left, omega_right):
        x, y, theta = state
        v = self.r / 2 * (omega_left + omega_right)  # linear speed
        delta = (omega_right - omega_left) / self.L  # angular speed
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = delta

        return np.array([dx, dy, dtheta])
    
    def estimate(self, omega_left, omega_right, dt):
        state = self.state.copy()
        k1 = self.state_function(state, omega_left, omega_right)
        k2 = self.state_function(state + 0.5 * dt * k1, omega_left, omega_right)
        k3 = self.state_function(state + 0.5 * dt * k2, omega_left, omega_right)
        k4 = self.state_function(state + dt * k3, omega_left, omega_right)
    
        # Update the state
        self.state += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return self.state






class KalmanFilter:
    def __init__(self, r, L, I_wheel, I_body, m_robot):
        """
        Initialize filter matrices for a differential drive robot with inertia and mass.

        @params:
            r: float, wheel radius
            L: float, wheelbase (distance between the wheels)
            I_wheel: float, moment of inertia of each wheel
            I_body: float, moment of inertia of the robot body
            m_robot: float, mass of the robot
        """
        # IMU noise standard deviations (adjust these as needed)
        self.acc_noise_std = 0.6  # Accelerometer noise standard deviation
        self.gyro_noise_std = 0.05  # Gyroscope noise standard deviation (example)
        self.r = r
        self.L = L
        self.I_wheel = I_wheel  # Moment of inertia of each wheel
        self.I_body = I_body    # Moment of inertia of the robot body
        self.m_robot = m_robot  # Mass of the robot
        self.P = np.eye(3)  # State covariance matrix (3x3) for [x, y, θ]
        self.Q = np.eye(3)  # Process noise covariance (simplified)
        self.R = np.array([[self.acc_noise_std**2, 0],
                   [0, self.gyro_noise_std**2]])
        self.state = np.zeros(3)  # Initial state [x, y, θ]

    def set_noise_matrices(self, Q, R):
        """Set custom noise matrices."""
        self.Q = Q
        self.R = R

    def predict(self, tau_r, tau_l, dt):
        """
        Predict the next state based on control inputs (motor torques) and system dynamics.

        @params:
            tau_r: float, right motor torque
            tau_l: float, left motor torque
            dt: float, time step
        """
        # Compute wheel angular accelerations from torques (taking inertia into account)
        alpha_r = tau_r / self.I_wheel  # Right wheel angular acceleration
        alpha_l = tau_l / self.I_wheel  # Left wheel angular acceleration

        # Compute the angular velocities of the wheels from angular accelerations
        omega_r = alpha_r * dt  # Right wheel angular velocity
        omega_l = alpha_l * dt  # Left wheel angular velocity

        # Compute the linear velocity and angular velocity of the robot
        v_r = omega_r * self.r  # Right wheel linear velocity
        v_l = omega_l * self.r  # Left wheel linear velocity

        v = (v_r + v_l) / 2  # Linear velocity (average of left and right wheel velocities)
        w = (v_r - v_l) / self.L  # Angular velocity (difference between wheels / wheelbase)

        # Current state
        x, y, theta = self.state

        # Predict new positions based on the velocities
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + w * dt

        # Update the state with new values
        self.state = np.array([x_new, y_new, theta_new])

        # State transition matrix F (approximated for a differential drive robot)
        F = np.eye(3)
        F[0, 2] = -v * np.sin(theta) * dt  # dx/dθ term
        F[1, 2] = v * np.cos(theta) * dt  # dy/dθ term

        # Update covariance matrix P
        self.P = F @ self.P @ F.T + self.Q

    def update(self, gyro, acc, dt):
        """
        Update the state based on noisy IMU measurements (gyro and acc).
        """
        # Simulate noise in the accelerometer and gyroscope measurements
        noisy_acc = acc + np.random.normal(0, self.acc_noise_std, len(acc))
        noisy_gyro = gyro + np.random.normal(0, self.gyro_noise_std, len(gyro))

        # Estimate v and w from IMU (noisy measurements)
        v = self.state[0] + noisy_acc[0] * dt  # Integrate noisy acceleration to estimate velocity
        w = noisy_gyro[2]  # Use noisy gyroscope angular velocity

        # Measurement vector (noisy measurements for v and w)
        z = np.zeros(2)
        z[0] = v
        z[1] = w

        # Measurement matrix H (mapping from state to measurement space)
        H = np.zeros((2, 3))
        H[0, 0] = 1  # Map linear velocity (v)
        H[1, 2] = 1  # Map angular velocity (w)

        # Kalman gain calculation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Innovation (difference between observed and predicted measurement)
        y = z - H @ self.state

        # Update state estimate
        self.state += K @ y

        # Update covariance estimate
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def get_state(self):
        """Return the current state (position and orientation)."""
        return self.state



        
