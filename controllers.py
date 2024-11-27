import math
import numpy as np
import logging

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
        theta = state["orientation"]  # Current orientation in radians
        x_d, y_d = self.params["x_d"], self.params["y_d"]
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

        # Compute control efforts
        w = K_p_theta * error_theta
        v = K_p_pos * error_pos

        # Scale linear velocity based on angular error
        v *= max(0, np.cos(error_theta))  # Reduce v as error_theta increases

        # Compute wheel velocities
        v_l = v - w
        v_r = v + w

        # Scale wheel velocities to maximum allowed speed
        max_wheel_speed = max(abs(v_l), abs(v_r))
        scale_factor = V_MAX / max_wheel_speed
        v_l *= scale_factor
        v_r *= scale_factor

        # Stop when we get close enough
        if error_pos < 0.25:
            v_l, v_r, = 0, 0

        return {"left_wheel": v_l, "right_wheel": v_r}


class ScheduledPIDController:
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.params = params

    def compute_controls(self, state, dt):
        x, y = state["position"]
        theta = state["orientation"]  # Current orientation in radians
        x_d, y_d = self.params["x_d"], self.params["y_d"]
        b = self.params["wheelbase"]
        K_p_theta = self.params["K_p_theta"]

        # Compute position error components
        delta_x = x_d - x
        delta_y = y_d - y

        # Compute desired heading
        theta_d = math.atan2(delta_y, delta_x)
        # Compute heading error relative to robot's orientation
        error_theta = theta - theta_d
        # Ensure wrapping to [-pi, pi]
        error_theta = (error_theta + math.pi) % (2 * math.pi) - math.pi

        # Compute Euclidean distance to target (position error)
        error_pos = math.hypot(delta_x, delta_y)

        # # Determine whether to move or turn in place
        # if abs(error_theta) > math.radians(90):  # If heading error > 90 degrees
        #     v = 0  # Stop linear motion
        #     w = K_p_theta * error_theta  # Rotate in place
        # else:
        #     v = 2  # Set constant linear velocity
        #     w = K_p_theta * error_theta  # Calculate angular velocity

        # # Saturate linear velocity (v) and angular velocity (w)
        # v = max(-1, min(1, v))  


        v = 10  # Set constant linear velocity
        w = K_p_theta * error_theta  # Calculate angular velocity

        # Compute wheel velocities
        v_r = v + w
        v_l = v - w

        # Scale wheel velocities to maximum allowed speed
        V_MAX = self.params["V_MAX"]
        max_wheel_speed = max(abs(v_r), abs(v_l))
        if max_wheel_speed > V_MAX:
            scale_factor = V_MAX / max_wheel_speed
            v_r *= scale_factor
            v_l *= scale_factor

        # Stop motion if position error is below threshold
        if error_pos < 1:  # Stop if close to target
            v_r, v_l = 0, 0

        return {"left_wheel": v_l, "right_wheel": v_r}





class SlidingModeController:
    def __init__(self, params):
        self.params = params

    def compute_controls(self, state, dt):
        x, y = state["position"]
        theta = state["orientation"]
        x_d, y_d = self.params["x_d"], self.params["y_d"]
        b = self.params["wheelbase"]

        # Compute errors
        e_x = x_d - x
        e_y = y_d - y
        e_pos = math.sqrt(e_x**2 + e_y**2)
        theta_d = math.atan2(e_y, e_x)
        e_theta = (theta - theta_d + math.pi) % (2 * math.pi) - math.pi
        # Sliding surfaces
        S_pos = e_pos
        S_theta = e_theta

        # Control laws
        v = -self.params["k_pos"] * S_pos / (self.params["epsilon"] + abs(S_pos))
        omega = -self.params["k_theta"] * S_theta / (self.params["epsilon"] + abs(S_theta))

        # Compute wheel velocities
        v_r = v + (b / 2.0) * omega
        v_l = v - (b / 2.0) * omega

        # # Limit wheel speeds
        # V_MAX = self.params["V_MAX"]
        # max_wheel_speed = max(abs(v_r), abs(v_l))
        # if max_wheel_speed > V_MAX:
        #     scale_factor = V_MAX / max_wheel_speed
        #     v_r *= scale_factor
        #     v_l *= scale_factor

        return {"left_wheel": v_l, "right_wheel": v_r}

