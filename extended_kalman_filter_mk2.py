import numpy as np
import matplotlib.pyplot as plt



def calc_new_position(X, U) :
    """ Calculates new position of robot using motion model given :
        current_x : _ meters 
        current_y : _ meters
        current_th : _ radlandmark_estimated_range
        trans_speed : _ meters/second 
        rot_speed : _ rad/second 
        time : _ seconds
    """
    current_x = X[0]
    current_y = X[1]
    current_th = X[2]

    trans_speed = U[0]
    rot_speed = U[1]


    pos_prev = np.array([current_x, current_y, current_th])
    ang_mat = np.array([[np.cos(current_th),0],[np.sin(current_th), 0], [0,1]])
    control = np.array([trans_speed, rot_speed])
    new_pos =  pos_prev + 0.1 * ang_mat @ control
    return new_pos


def observation_model(pos, landmark_true_pos, d, obs_i):
    """
    return 34*1
    take landmark_true_pos as all the landmarks
    landmark_true_pos = 17*2
    """
    output = np.zeros((34,1))
    for i in range(landmark_true_pos.shape[0]) :
        if obs_i[2*i] == 0:
            output[2*i] = 0
            output[2*i + 1] = 0
        else:
                
            output[2*i] = np.sqrt((landmark_true_pos[i,0]-pos[0]-d*np.cos(pos[2]))**2 + (landmark_true_pos[i,1]-pos[1]-d*np.sin(pos[2]))**2)
            output[2*i+1] = np.arctan2(landmark_true_pos[i,1]-pos[1]-d*np.sin(pos[2]),landmark_true_pos[i,0]-pos[0]-d*np.cos(pos[2])) - pos[2]
            while output[2*i + 1] >= np.pi :
                output[2*i + 1] -= 2*np.pi
            while output[2*i + 1] <= -np.pi:
                output[2*i + 1] += 2*np.pi

    return output

    
def ret_sensor_jacobian_l(x_t, r_l, d, obs_i):
    """
    calculates jacobian for lth landmark
    at timestep t
    Arguments: d, current belief, and landmark position
    return 2*3 jacobian
    """
    if obs_i == 0:
        return np.zeros((2,3))

    x_k = x_t[0]
    y_k = x_t[1]
    theta_k = x_t[2]
    x_l = r_l[0]
    y_l = r_l[1]
    a = x_l-x_k-d*np.cos(theta_k)
    b = y_l - y_k - d*np.sin(theta_k)
    t = np.sqrt(a*a + b*b)
    j_00 = -1 * a / t
    j_01 = -1 * b / t
    j_02 = (1/t) * (a*d*np.sin(theta_k) - b*d*np.cos(theta_k))
    j_10 = b/(t*t)
    j_11 = -a/(t*t)
    j_12 = -d * (a * np.cos(theta_k) + b*np.sin(theta_k))/(t*t) - 1 
    jacob = np.array([[j_00,j_01,j_02],[j_10,j_11,j_12]])
    return jacob


def ret_sensor_jacobian(x_t, x_l, y_l, d, obs):
    for i in range(17):
        r_l = np.array([x_l[i], y_l[i]])
    
        if i == 0:
            big_jacob = ret_sensor_jacobian_l(x_t, r_l, d, obs[2*i])
        else:
            H = ret_sensor_jacobian_l(x_t, r_l, d, obs[2*i])
            big_jacob = np.concatenate((big_jacob, H), axis = 0)
    return big_jacob





def ret_motion_jacobian(prev_th, trans_speed):
    """ Motion Jacobian only dependant on trans_speed and prev_th, 
    trans_speed : _ meters/second
    prev_th : _ rad
    """
    motion_jacobian = np.zeros((3,3))
    I = np.eye(3)
    term_2 = np.zeros((3,3))
    term_2[0,2] = -trans_speed * np.sin(prev_th)
    term_2[1,2] = trans_speed * np.cos(prev_th)
    dt = 0.1
    motion_jacobian = I + dt * term_2
    
    return motion_jacobian


    

def ekf(prev_pos, prev_cov, control, obs, d, Rt, Qt, landmarks):
    """ Takes prev position, prev_cov, current control ,and current obs """
    G = ret_motion_jacobian(prev_th=prev_pos[2],trans_speed=control[0])
    x_l = landmarks[:,0]
    y_l = landmarks[:,1]
    H = ret_sensor_jacobian(prev_pos, x_l, y_l, d, obs)
    pos_current_dash = calc_new_position(prev_pos, control)
    # pos_current_dash = calc_new_pos(current_control,prev_pos)
    current_cov_dash = G@prev_cov@G.T + Rt
    kalman_gain = current_cov_dash@H.T@np.linalg.inv(H@current_cov_dash@H.T + Qt)
    """
    replacing h here
    """
    h = observation_model(pos_current_dash, landmarks, d, obs)
    # pos_current = pos_current_dash + kalman_gain@(obs - h(pos_current_dash))
    h.shape = 34
    pos_current = pos_current_dash + kalman_gain@(obs - h)
    cov_current = (np.eye(3) - kalman_gain@H)@current_cov_dash
    
    return pos_current, cov_current
 

def return_data(dataset):
    #Rt
    v_var = dataset['v_var'][0][0]
    om_var = dataset['om_var'][0][0]
    Rt = np.diag([v_var, v_var, om_var])
    #Qt
    r_var = dataset['r_var'][0][0]
    b_var = dataset['b_var'][0][0]
    Qt = np.zeros((34,34))
    for i in range(34):
        if i%2 == 0:
            Qt[i][i] = r_var 
        else:
            Qt[i][i] = b_var

    #Robot position
    x_true = dataset['x_true']
    y_true = dataset['y_true']
    th_true = dataset['th_true']

    X = np.concatenate((x_true, y_true), axis = 1)
    X = np.concatenate((X, th_true), axis = 1)
    l = dataset['l']
    r = dataset['r']
    b = dataset['b']
    v = dataset['v']
    om = dataset['om']
    d = dataset['d'][0][0]
    u = np.concatenate((v, om), axis=1)
    return Rt, Qt, X, l, r, b, u, d


def plot_ground_truth(X, l, steps, col): 
    fig = plt.figure(figsize=(10,5))
    plt.scatter(l[:,0], l[:,1], color = 'r', marker='x')
    plt.plot(X[0:steps,0], X[0:steps,1])

def plot_motion_model(X, u, steps):
    new_pos = []
    new_pos.append(X)
    for i in range(1, steps):
        temp = calc_new_position(new_pos[i-1] ,u[i])
        new_pos.append(temp)

    return np.array(new_pos)



def ret_obs(r_i, b_i):
    obs = np.zeros(34)
    for i in range(17):
        obs[2*i] = r_i[i]
        obs[2*i+1] = b_i[i]
    return obs


def ekf_predicted(X_init, r, b, u, l, d, Rt, Qt, steps):
    x_pred = []
    x_pred.append(X_init)
    prev_cov = np.diag([1,1,0.1])
    for i in range(1, steps):
        prev_pos = x_pred[i-1]
        control = u[i]
        r_i = r[i]
        b_i = b[i]
        obs = ret_obs(r_i, b_i)
        landmarks = l
        prev_pos, prev_cov = ekf(prev_pos, prev_cov, control, obs, d, Rt, Qt, landmarks)
        x_pred.append(prev_pos)
    
    return np.array(x_pred)

dataset = np.load('dataset.npz')
Rt, Qt, X, l, r, b, u, d = return_data(dataset)
steps = 2000
X_init = X[0]


x_motion = plot_motion_model(X_init, u, steps)


x_pred = ekf_predicted(X_init, r, b, u, l, d, Rt, Qt, steps)
#Plot using ONLY wheel odometry, ground truth and landmark locations

a = x_motion[0:steps,0]
b = x_motion[0:steps,1]
c = X[0:steps,0]
d = X[0:steps,1]
# d = c[::-1]
plt.style.use('ggplot')
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, b,  label='Wheel odometry')
ax.plot(c, d,  label='Ground Truth')
ax.scatter(l[:,0], l[:,1], label='landmarks', marker='x')
ax.set_title("Wheel odometry and ground truth")
legend = ax.legend(loc='upper right', fontsize='x-large', fancybox=True, framealpha=0.9)

# Put a nicer background color on the legend.
plt.show()




a = x_pred[0:steps,0]
b = x_pred[0:steps,1]
c = X[0:steps,0]
d = X[0:steps,1]
# d = c[::-1]

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, b,  label='EKF-estimated')
ax.plot(c, d,  label='Ground Truth')
ax.scatter(l[:,0], l[:,1], label='landmarks', marker='x')
ax.set_title("EKF-estimated and ground truth")
legend = ax.legend(loc='upper right', fontsize='x-large', fancybox=True, framealpha=0.9)

# Put a nicer background color on the legend.
plt.show()





# plt.plot(x_motion[0:steps,0], x_motion[0:steps,1], color = 'b')
# plt.plot(X[0:steps,0], X[0:steps,1], color = 'r')
# plt.scatter(l[:,0], l[:,1], marker='x')
# plt.title("Ground truth vs wheel odometry")
# plt.show()

# plt.plot(x_pred[0:steps,0], x_pred[0:steps,1], color = 'g')
# plt.show()