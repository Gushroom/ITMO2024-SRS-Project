import math
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
        theta = state["orientation"]
        x_d, y_d = self.params["x_d"], self.params["y_d"]

        def uniToDiff(self, v, w):
            vR = (2*v + w*self.L)/(2*self.R)
            vL = (2*v - w*self.L)/(2*self.R)
            return vR, vL
        
        def diffToUni(self, vR, vL):
            v = self.R/2*(vR+vL)
            w = self.R/self.L*(vR-vL)
            return v, w

        # Compute errors
        e_x = x_d - x
        e_y = y_d - y
        g_theta = math.atan2(e_y, e_x)
        alpha = -(g_theta - theta)
        e = math.atan2(math.sin(alpha), math.cos(alpha))

        e_P = e
        # e_I = self.integral_e_theta + e
        # e_D = e - self.e_theta_prev

        K_p_theta = self.params["K_p_theta"] 
        
        w = K_p_theta * e_P

        w = math.atan2(math.sin(w), math.cos(w))

        self.e_theta_prev = e 

        v = 0

        # Compute wheel velocities
        # b = self.params["wheelbase"]

        # def fixAngle(self, angle):
        #     return math.atan2(math.sin(angle), math.cos(angle))
        
        # def makeAction(self, v, w):
        #     x_dt = v*math.cos(theta)



        # # Limit wheel speeds
        # V_MAX = self.params["V_MAX"]
        # max_wheel_speed = max(abs(v_r), abs(v_l))
        # if max_wheel_speed > V_MAX:
        #     scale_factor = V_MAX / max_wheel_speed
        #     v_r *= scale_factor
        #     v_l *= scale_factor

        # return {"left_wheel": v_l, "right_wheel": v_r}

        b = self.params["wheelbase"]

        vR = (2*v + w*b)/(2*0.2)
        vL = (2*v - w*b)/(2*0.2)
        return {"left_wheel": vL, "right_wheel": vR}
    

class ScheduledPIDController:
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

        # Transform target to robot's local frame
        e_x = x_d - x
        e_y = y_d - y
        x_local = math.cos(theta) * e_x + math.sin(theta) * e_y
        y_local = -math.sin(theta) * e_x + math.cos(theta) * e_y

        # Compute desired heading in local frame
        theta_d = math.atan2(y_local, x_local)  # Heading error in local frame

        # Compute position error (distance to target)
        e_pos = math.sqrt(x_local**2 + y_local**2)
        # Compute heading error
        e_theta = theta_d - theta

        # Log for debugging
        logging.info(f"Position: ({x:.2f}, {y:.2f}), "
                    f"Orientation: {math.degrees(theta):.2f}°, "
                    f"Local Target: ({x_local:.2f}, {y_local:.2f}), "
                    f"Heading Error: {math.degrees(theta_d):.2f}°")


        K_p_theta = self.params["K_p_theta"] 

        K_p_pos = self.params["K_p_pos"]

        b = self.params["wheelbase"]
        # heading_error_threshold = math.radians(10)
        # moving_error_threshold = math.radians(30)
        # if abs(e_theta) > heading_error_threshold:
        #     # correct heading first
        #     K_p_pos = 0
        #      # Enable turning
        # else:
        #     K_p_theta = 0
        #       # Enable moving forward


        # Compute control signals
        v = -(K_p_pos * e_pos) # Linear velocity
        omega = K_p_theta * theta_d  # Angular velocity

        # Compute wheel velocities
        b = self.params["wheelbase"]
        v_r = ((2 * v + omega * b) / (2 * 0.2)) / 0.2
        v_l = ((2 * v - omega * b) / (2 * 0.2)) / 0.2

        # Limit wheel speeds
        V_MAX = self.params["V_MAX"]
        max_wheel_speed = max(abs(v_r), abs(v_l))
        if max_wheel_speed > V_MAX:
            scale_factor = V_MAX / max_wheel_speed
            v_r *= scale_factor
            v_l *= scale_factor

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

