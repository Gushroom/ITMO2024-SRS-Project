from scipy.integrate import quad
import numpy as np
from scipy.linalg import expm



def create_velocity_model(b, m, r, L, I,Iw ,Ts,d, mc,sigma_r, sigma_l,state):
    wk = state[2]
    vk = state[3]
    a12 = float(- (mc * d* wk *Ts)/ (I + 2*L**2 * Iw/r**2))
    a11 = float(1 - (mc * d* vk *Ts)/ (I + 2*L**2 * Iw/r**2))
    a22 = 1.0
    a21 = float((2* mc * d *wk * Ts) / (m + 2*Iw/r**2))

    print(f"({float(a11)}, {a12}, {a21}, {a22})")

    b21 = Ts/(m + 2*Iw/r**2)/r 
    b22  =Ts/(m + 2*Iw/r**2)/r 
    b11 = L* Ts/r /(I + 2*L**2 * Iw/r**2)
    b12 = - L* Ts/r /(I + 2*L**2 * Iw/r**2) 

    A_dis = np.array([[0,0,1,0],[0,0,0,1],[0, 0, a11, a12],[0, 0, a21, a22]])
    B_dis = np.array ([[0,0],[0, 0],[b11, b12],[b21, b22]])
    Sigma_matrix = np.array([[sigma_r**2, 0],[0, sigma_l**2]])
    Noise_Cov = B_dis @ Sigma_matrix @ B_dis.T
    return A_dis, B_dis, Noise_Cov

def create_measurement_model (Ts,sigma_a, sigma_w):
    H = np.array([[-1/Ts, 0, 1/Ts, 0],[0, 0, 0, 1]])
    Noise_Cov = np.array([[sigma_a**2, 0],[0, sigma_w**2]])
    return H, Noise_Cov




def motion_model (v, w, x, y, theta, Ts):
    if w == 0.0:
        w_new = 1e-30
    else:
        w_new = w
    theta_new = theta + w*Ts
    x_new = x + (-v/w_new) * np.sin(theta) + (v/w_new) * np.sin(theta_new)
    y_new = y + (v/w_new) * np.cos(theta) - (v/w_new) * np.cos(theta_new)

    return x_new, y_new, theta_new
     
def Kalman_filter_step(A,B,Q,H,R,x_last,u,z,Cov,Ts,I,Iw,r,mc,L,m,d,correction_flag = True,prediction_flag = True):
    if prediction_flag == False:
        x_new = np.array([[float(x_last[2])],[float(x_last[3])],[0],[0]])
        x_new[2] = z[1]
        x_new[3] = z[0]*Ts +x_last[3]
        return x_new, Cov

    
    x_prediction = np.array([[float(x_last[2])],[float(x_last[3])],[0],[0]])
    x_prediction[2] = float(((L/r)*(u[0]-u[1]) - mc*d*x_last[2]*x_last[3]) * Ts/(I + 2*L**2 * Iw/r**2) + x_last[2])
    x_prediction[3] = float(((1/r)*(u[0]+u[1]) + mc*d*x_last[2]**2) * Ts/(m + 2* Iw/r**2) + x_last[3])
    Cov_prediction = A @ Cov @ A.T + Q 
    if correction_flag:
        S = H @ Cov_prediction @ H.T + R
        K = Cov_prediction @ H.T @ np.linalg.inv(S)

        x_new = x_prediction + K @ (z - H @ x_prediction)
        Cov_new = (np.identity(4) - K @ H) @ Cov_prediction
        return x_new, Cov_new
    else:
        return x_prediction, Cov_prediction

    

    
