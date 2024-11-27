from scipy.integrate import quad
import numpy as np
from scipy.linalg import expm



def create_velocity_model(b, m, r, L, I, Ts, sigma_r, sigma_l):

    A_dis = np.array([[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 0, 0, np.exp(-b*Ts/m)]])
    B_dis = np.array ([[0,0],[0, 0],[(m*L*Ts)/(2*I), -(m*L*Ts)/(2*I)],[r*m*(1-np.exp(-b*Ts/m))/(2*b), r*m*(1-np.exp(-b*Ts/m))/(2*b)]])
    Sigma_matrix = np.array([[sigma_r**2, 0],[0, sigma_l**2]])
    Noise_Cov = B_dis @ Sigma_matrix @ B_dis.T
    return A_dis, B_dis, Noise_Cov

def create_measurement_model (Ts,sigma_a, sigma_w):
    H = np.array([[-1/Ts, 0, 1/Ts, 0],[0, 0, 0, 1]])
    Noise_Cov = np.array([[sigma_a**2, 0],[0, sigma_w**2]])
    return H, Noise_Cov

A,B,Q = create_velocity_model(0.1,2,0.3,0.4,0.1,0.01,0.1,0.2)
H, Q = create_measurement_model(0.1,0.1,0.2)


def motion_model (v, w, x, y, theta, Ts):
    if w == 0.0:
        w_new = 1e-30
    else:
        w_new = w
    theta_new = theta + w*Ts
    x_new = x + (-v/w_new) * np.sin(theta) + (v/w_new) * np.sin(theta_new)
    y_new = y + (v/w_new) * np.cos(theta) - (v/w_new) * np.cos(theta_new)

    return x_new, y_new, theta_new
     
def Kalman_filter_step(A,B,Q,H,R,x_last,u,z,Cov,correction_flag = True):
    x_prediction = A @ x_last + B @ u
    Cov_prediction = A @ Cov @ A.T + Q 
    if correction_flag:
        S = H @ Cov_prediction @ H.T + R
        K = Cov_prediction @ H.T @ np.linalg.inv(S)

        x_new = x_prediction + K @ (z - H @ x_prediction)
        Cov_new = (np.identity(4) - K @ H) @ Cov_prediction
    else:
        return x_prediction, Cov_prediction

    return x_new, Cov_new

    
