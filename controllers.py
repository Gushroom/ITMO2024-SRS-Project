import numpy as np
import logging


def angleSubtraction(angle_a, angle_b):
    diff = angle_a - angle_b
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

class VelocityPID():
    def __init__(self, params):
        self.params = params
        self.e_pos_prev = 0.0
        self.e_theta_prev = 0.0
        self.integral_e_pos = 0.0
        self.integral_e_theta = 0.0

    def compute_controls(self, state, x_d, y_d, dt):
        x, y = state["position"]
        theta = state["orientation"]  # Current orientation in radians

        K_p_theta = self.params["K_p_theta"]
        K_i_theta = self.params["K_i_theta"]
        K_d_theta = self.params["K_d_theta"]
        K_p_pos_base = self.params["K_p_pos"]
        K_i_pos = self.params["K_i_pos"]
        K_d_pos = self.params["K_d_pos"]
        base = self.params["wheelbase"]
        V_MAX = self.params["V_MAX"]

        # Compute desired heading and position error
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        theta_d = np.atan2(y_d - y, x_d - x)
        theta_d = (theta_d + np.pi) % (2 * np.pi) - np.pi
        error_theta = angleSubtraction(theta_d, theta)
        error_pos = np.hypot(y_d - y, x_d - x)

        # Scale linear velocity based on angular error
        k = 10  # Steepness
        theta_threshold = np.pi / 4  # Threshold (45 deg)
        K_p_pos = K_p_pos_base / (1 + np.exp(k * (abs(error_theta) - theta_threshold))) # Reduce linear gain as error_theta increases

        # K_p_pos = K_p_pos_base * max(0, np.cos(error_theta))

        # K_p_pos = K_p_pos_base

        # Compute control efforts
        # Update integral and derivative terms
        self.integral_e_pos += error_pos * dt
        self.integral_e_theta += error_theta * dt
        derivative_e_pos = (error_pos - self.e_pos_prev) / dt
        derivative_e_theta = (error_theta - self.e_theta_prev) / dt

        # Compute control efforts
        w = (K_p_theta * error_theta +
             K_i_theta * self.integral_e_theta +
             K_d_theta * derivative_e_theta)
        v = (K_p_pos * error_pos +
             K_i_pos * self.integral_e_pos +
             K_d_pos * derivative_e_pos)

        # Save errors for next iteration
        self.e_pos_prev = error_pos
        self.e_theta_prev = error_theta

        # Compute wheel velocities
        v_l = v - w * (base / 2.0)
        v_r = v + w * (base / 2.0)

        # Scale wheel velocities to maximum allowed speed
        max_wheel_speed = max(abs(v_l), abs(v_r))
        scale_factor = V_MAX / max_wheel_speed
        v_l *= scale_factor
        v_r *= scale_factor

        return {"left_wheel": v_l, "right_wheel": v_r}
    
class AccelerationPID():
    def __init__(self, params):
        self.params = params
        self.prev_vl = 0
        self.prev_vr = 0
        self.err_vl_prev = 0
        self.err_vr_prev = 0
        self.integral_err_vl = 0
        self.integral_err_vr = 0
        self.derivative_err_vl_prev = 0
        self.derivative_err_vr_prev = 0

    def compute_controls(self, desired_velocity, actural_velocity, dt):
        K_p = self.params["K_p"]
        K_i = self.params["K_i"]
        K_d = self.params["K_d"]
        T_MAX = self.params["T_MAX"]  # max control limit (10)
        T_MIN = self.params["T_MIN"]  # min control limit (-10)
        r = self.params["wheelradius"]

        # # Unpack velocities and calculate actural v_l and v_r using v, w, base
        # actural_linear = actural_velocity[0]
        # actural_angular = actural_velocity[1]
        # actural_vl = actural_linear - (base/2.0) * actural_angular
        # actural_vr = actural_linear + (base/2.0) * actural_angular
        omega_left, omega_right = actural_velocity
        actural_vl = omega_left * r
        actural_vr = omega_right * r

        desired_vl = desired_velocity[0]
        desired_vr = desired_velocity[1]

        # Compute errors between desired and current velocities
        err_left = desired_vl - actural_vl
        err_right = desired_vr - actural_vr
        # Update integral of errors
        MAX_INTEGRAL = 10.0
        self.integral_err_vl += err_left * dt
        self.integral_err_vr += err_right * dt
        # Anti-windup
        self.integral_err_vl = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.integral_err_vl))
        self.integral_err_vr = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.integral_err_vr))

        # Compute derivative of errors
        derivative_err_vl = (err_left - self.err_vl_prev) / dt
        derivative_err_vr = (err_right - self.err_vr_prev) / dt
        # Low pass filter to smooth D term
        ALPHA = 0.1 
        derivative_err_vl = ALPHA * derivative_err_vl + (1 - ALPHA) * self.derivative_err_vl_prev
        derivative_err_vr = ALPHA * derivative_err_vr + (1 - ALPHA) * self.derivative_err_vr_prev
        self.derivative_err_vl_prev = derivative_err_vl
        self.derivative_err_vr_prev = derivative_err_vr


        # PID control laws for left and right wheel torques
        tau_l = K_p * err_left + K_i * self.integral_err_vl + K_d * derivative_err_vl
        tau_r = K_p * err_right + K_i * self.integral_err_vr + K_d * derivative_err_vr

        # Clamp the torques to within the allowed limits
        tau_l = max(T_MIN, min(T_MAX, tau_l))
        tau_r = max(T_MIN, min(T_MAX, tau_r))

        # Update previous errors for the next iteration
        self.err_vl_prev = err_left
        self.err_vr_prev = err_right

        # Return left and right wheel motor torques
        return {"tau_left": tau_l, "tau_right": tau_r}
    

class VelocitySlidingMode():
    def __init__(self, params):
        self.params = params
        self.e_x_prev = 0.0
        self.e_y_prev = 0.0
        self.e_theta_prev = 0.0

    def compute_controls(self, state, x_d, y_d, dt):
        x, y = state["position"]
        theta = state["orientation"]  # Current orientation in radians
        V_MAX = self.params["V_MAX"]  # Maximum velocity
        base = self.params["wheelbase"]

        # Sliding mode control parameters
        K_pos_base = self.params["K_pos"]
        K_theta = self.params["K_theta"]
        lambda_pos = self.params["lambda_pos"]
        lambda_theta = self.params["lambda_theta"]
        epsilon_pos = self.params["epsilon_pos"]
        epsilon_theta = self.params["epsilon_theta"]

        # Normalize angles
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        theta_d = np.arctan2(y_d - y, x_d - x)
        theta_d = (theta_d + np.pi) % (2 * np.pi) - np.pi

        # Compute errors
        error_x = x_d - x
        error_y = y_d - y
        error_theta = angleSubtraction(theta_d, theta)

        # Sliding surfaces
        s_x = error_x + lambda_pos * (error_x - self.e_x_prev) / dt
        s_y = error_y + lambda_pos * (error_y - self.e_y_prev) / dt
        s_theta = error_theta + lambda_theta * (error_theta - self.e_theta_prev) / dt

        # Update previous errors
        self.e_x_prev = error_x
        self.e_y_prev = error_y
        self.e_theta_prev = error_theta

        # K_pos = K_pos_base * max(0, np.cos(error_theta))

        # Control laws with smoothing to avoid chattering
        alpha, beta = 0.1, 0.3
        v = K_pos_base * np.tanh((s_x * np.cos(theta) + s_y * np.sin(theta)) / epsilon_pos) + alpha * np.sign(s_x * np.cos(theta) + s_y * np.sin(theta))
        w = K_theta * np.tanh(s_theta / epsilon_theta) + beta * np.sign(s_theta)

        # w = K_theta * np.tanh(s_theta / epsilon_theta)

        # Compute wheel velocities
        v_l = v - w * (base / 2.0)
        v_r = v + w * (base / 2.0)

        # Scale wheel velocities to maximum allowed speed
        max_wheel_speed = max(abs(v_l), abs(v_r))
        scale_factor = V_MAX / max_wheel_speed
        v_l *= scale_factor
        v_r *= scale_factor

        return {"left_wheel": v_l, "right_wheel": v_r}






