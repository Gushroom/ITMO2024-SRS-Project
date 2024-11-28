import numpy as np
import logging

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
        K_p_pos = self.params["K_p_pos"]
        V_MAX = self.params["V_MAX"]  # Maximum velocity

        def angleSubtraction(angle_a, angle_b):
            diff = angle_a - angle_b
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            return diff

        # Compute desired heading and position error
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        theta_d = np.atan2(y_d - y, x_d - x)
        theta_d = (theta_d + np.pi) % (2 * np.pi) - np.pi
        error_theta = angleSubtraction(theta_d, theta)
        error_pos = np.hypot(y_d - y, x_d - x)

        # Scale linear velocity based on angular error
        K_p_pos *= max(0, np.cos(error_theta))  # Reduce linear gain as error_theta increases

        # Compute control efforts
        w = K_p_theta * error_theta
        v = K_p_pos * error_pos

        # Compute wheel velocities
        v_l = v - w
        v_r = v + w

        # Scale wheel velocities to maximum allowed speed
        max_wheel_speed = max(abs(v_l), abs(v_r))
        scale_factor = V_MAX / max_wheel_speed
        v_l *= scale_factor
        v_r *= scale_factor

        # Stop when we get close enough
        if error_pos < 0.1:
            v_l, v_r, = 0, 0

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

    def compute_controls(self, desired_velocity, actural_velocity, dt):
        K_p = self.params["K_p"]
        K_i = self.params["K_i"]
        K_d = self.params["K_d"]
        T_MAX = self.params["T_MAX"]  # max control limit (30)
        T_MIN = self.params["T_MIN"]  # min control limit (-30)
        base = self.params["wheelbase"]

        # Unpack velocities and calculate actural v_l and v_r using v, w, base
        actural_linear = actural_velocity[0][0]
        actural_angular = actural_velocity[1][0]
        actural_vl = actural_linear - (base/2.0) * actural_angular
        actural_vr = actural_linear + (base/2.0) * actural_angular

        desired_vl = desired_velocity[0]
        desired_vr = desired_velocity[1]

        # Compute errors between desired and current velocities
        err_left = desired_vl - actural_vl
        err_right = desired_vr - actural_vr
        # Update integral of errors
        self.integral_err_vl += err_left * dt
        self.integral_err_vr += err_right * dt

        # Compute derivative of errors
        derivative_err_vl = (err_left - self.err_vl_prev) / dt
        derivative_err_vr = (err_right - self.err_vr_prev) / dt

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






