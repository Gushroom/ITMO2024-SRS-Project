import math
import numpy as np

class PIDController():
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.params = params
        self.e_pos_prev = 0.0
        self.e_theta_prev = 0.0
        self.integral_e_pos = 0.0
        self.integral_e_theta = 0.0

    def compute_controls(self, state, dt):
        x, y = state["position"]
        theta = state["orientation"]
        x_d, y_d = self.params["x_d"], self.params["y_d"]

        # Compute errors
        e_x = x_d - x
        e_y = y_d - y
        e_pos = math.sqrt(e_x**2 + e_y**2)
        # Desired theta is found by error in position
        theta_d = math.atan2(e_y, e_x)
        e_theta = (theta_d - theta + math.pi) % (2 * math.pi) - math.pi

        # Update integral and derivative terms
        self.integral_e_pos += e_pos * dt
        self.integral_e_theta += e_theta * dt
        derivative_e_pos = (e_pos - self.e_pos_prev) / dt
        derivative_e_theta = (e_theta - self.e_theta_prev) / dt

        # Compute PID outputs
        v = (self.params["K_p_pos"] * e_pos +
             self.params["K_i_pos"] * self.integral_e_pos +
             self.params["K_d_pos"] * derivative_e_pos)

        omega = (self.params["K_p_theta"] * e_theta +
                 self.params["K_i_theta"] * self.integral_e_theta +
                 self.params["K_d_theta"] * derivative_e_theta)

        # Update previous errors
        self.e_pos_prev = e_pos
        self.e_theta_prev = e_theta

        # Compute wheel velocities
        b = self.params["wheelbase"]
        v_r = v + (b / 2.0) * omega
        v_l = v - (b / 2.0) * omega

        # Limit wheel speeds
        V_MAX = self.params["V_MAX"]
        max_wheel_speed = max(abs(v_r), abs(v_l))
        if max_wheel_speed > V_MAX:
            scale_factor = V_MAX / max_wheel_speed
            v_r *= scale_factor
            v_l *= scale_factor

        return {"left_wheel": v_l, "right_wheel": v_r}
