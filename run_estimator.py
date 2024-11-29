import time
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
import kalman_estimator


model = mujoco.MjModel.from_xml_path('modified_robot.xml')
data = mujoco.MjData(model)

left_wheel_values = []
right_wheel_values = []
t_values = []
accel_values = []
gyro_values = []
x_true_values = []
y_true_values = []
theta_true_valeus = []
x_est_values = []
y_est_values = []
theta_est_valeus = []
left_torque_values = []
right_torque_values = []
v_est_values = []
w_est_values = []
v_true_values = []
w_true_values = []

initial_state_v = np.array([[0], [0], [0], [0]])
initial_cov = np.array([[0.1, 0, 0, 0],[0, 0.1, 0, 0],[0, 0, 0.1, 0],[0, 0, 0, 0.1]])
state_v_prev = initial_state_v
cov_prev = initial_cov
initial_pose = np.array([[0],[0],[0]])
prev_pose = initial_pose


loop_counter = 1
Ts = loop_counter*model.opt.timestep

H, Noise_cov_mes = kalman_estimator.create_measurement_model(Ts= Ts,sigma_a= 0.001,sigma_w= 0.001)
simulation_time = 50


with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    update_time = -np.inf
    while viewer.is_running() and time.time() - start < simulation_time:
        step_start = time.time()
        
        if loop_counter==1:
            loop_counter = 0

            
            #joints values
            left_wheel_values.append(float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT, 'left_wheel')]]))
            right_wheel_values.append(float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT, 'right_wheel')]]))

            #Accelerometer values
            accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_accelerometer')
            start_idx_accel = model.sensor_adr[accel_id]  
            accel_data = data.sensordata[start_idx_accel:start_idx_accel + 3]  
            #Gyro values
            gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
            start_idx_gyro = model.sensor_adr[gyro_id]  
            gyro_data = data.sensordata[start_idx_gyro:start_idx_gyro + 3]  
            accel_values.append( accel_data)   
            gyro_values.append( gyro_data)  

            # actuator values
            left_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_velocity_servo')
            right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_velocity_servo')
            left_torque_values.append(data.actuator_force[left_actuator_id])
            right_torque_values.append(data.actuator_force[right_actuator_id])

            #true pose values
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_body')
            position = data.xpos[body_id]
            x_true_values.append(position[0])
            y_true_values.append(position[1])
            orientation_quat = data.xquat[body_id]
            w, x, y, z = orientation_quat
            theta_true_valeus.append(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2)))

            # The control input for the velocity model (left and right acceleration)
            u = np.array([[data.ctrl[right_actuator_id]],[data.ctrl[left_actuator_id]]])
            # The measurments (forward acceleration and angular velocity from the IMU)
            measurement = np.array([[accel_data[0]],[gyro_data[2]]])
            # Kalman step gives us the estimated linear and angular velocity
            A, B, Noise_cov_sys = kalman_estimator.create_velocity_model(b = 0.0005,m = 1.6, r = 0.2, L = 0.2, I = 0.046804, Ts = Ts, sigma_r= 0.1, sigma_l= 0.1,Iw= 0.004,mc = 1.2, d = 0.2,state=state_v_prev)
            state_v, cov = kalman_estimator.Kalman_filter_step(A=A,B=B,Q= Noise_cov_sys,H=H,R=Noise_cov_mes, x_last = state_v_prev,u = u,z = measurement,Cov= cov_prev,correction_flag=False,prediction_flag=True,Ts=Ts,m=1.6,r=0.2,L=0.2,I=0.046804,Iw=0.004,mc =1.2,d=0.2)
            state_v_prev = state_v
            cov_prev = cov
            v_est_values.append(state_v[3])
            w_est_values.append(state_v[2])

            #get the true linear and angular velocity from mujoco
            qvel_start_index = model.body_dofadr[body_id]
            linear_velocity = data.qvel[qvel_start_index : qvel_start_index + 3]
            angular_velocity = data.qvel[qvel_start_index + 3:qvel_start_index +  6]
            base_link_id = model.body(body_id).id
            # Extract the rotation matrix (3x3) of the base link with respect to the world
            rotation_matrix_flat = data.xmat[base_link_id].reshape(3, 3)
            linear_velocity = np.array(linear_velocity).reshape(3,1)
            linear_velocity = rotation_matrix_flat.T @ linear_velocity
            v_true_values.append(linear_velocity[0])
            w_true_values.append(angular_velocity[2])
        
            # get the estimated pose (x,y,theta) from the motion model using the estimated linear and angular velocity
            x_est, y_est, theta_est = kalman_estimator.motion_model(state_v[3],state_v[2],prev_pose[0],prev_pose[1],prev_pose[2],Ts)
            prev_pose = [x_est,y_est,theta_est]
            x_est_values.append(x_est)
            y_est_values.append(y_est)
            theta_est_valeus.append(theta_est)

            print(f"acceleraction: {accel_data}")



            t_values.append(float(time.time() - start))
        


        loop_counter +=1
        # Step simulation
        mujoco.mj_step(model, data)
        # Update viewer settings
        
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


print(Ts)
plt.figure()
plt.plot(t_values,v_true_values)
plt.plot(t_values,v_est_values)
plt.ylabel("velocity (m/s)")
plt.xlabel("time (s)")
plt.title("Linear velocity value")
plt.grid(True)
plt.legend(["true_value","estimated"])
plt.show()


plt.figure()
plt.plot(t_values,w_true_values)
plt.plot(t_values,w_est_values)
plt.ylabel("angular velocity (rad/s)")
plt.xlabel("time (s)")
plt.title("angular value value")
plt.grid(True)
plt.legend(["true_value","estiamted"])
plt.show()

plt.figure()
plt.plot(t_values,x_true_values)
plt.plot(t_values,x_est_values)
plt.ylabel("Distance (m)")
plt.xlabel("time (s)")
plt.title("X value")
plt.grid(True)
plt.legend(["true_value","estimated"])
plt.show()

plt.figure()
plt.plot(t_values,y_true_values)
plt.plot(t_values,y_est_values)
plt.ylabel("Distance (m)")
plt.xlabel("time (s)")
plt.title("y value")
plt.grid(True)
plt.legend(["true_value","estiamted"])
plt.show()

plt.figure()
plt.plot(t_values,theta_true_valeus)
plt.plot(t_values,theta_est_valeus)
plt.ylabel("angle (rad)")
plt.xlabel("time (s)")
plt.title("theta value")
plt.grid(True)
plt.legend(["true_value","estiamted"])
plt.show()


