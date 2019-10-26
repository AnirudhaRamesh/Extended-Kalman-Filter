import numpy as np
import matplotlib.pyplot as plt

def read_data(filename) :
    """ Function to read data from given input dataset with the format of "Lost in the woods" dataset. """
    dataset = np.load(filename)
    # secs
    timestamps = dataset['t']
    # meters
    robot_true_x = dataset['x_true']
    robot_true_x.shape = (12609)

    robot_true_y = dataset['y_true']
    robot_true_y.shape = (12609)

    robot_true_th = dataset['th_true']
    robot_true_th.shape = (12609)

    landmark_true_pos = dataset['l']
    landmark_estimated_range = dataset['r']
    # meter^2
    landmark_estimated_range_var = dataset['r_var'][0][0]
    # rad
    landmark_estimated_bearing = dataset['b']
    # rad^2
    landmark_estimated_bearing_var = dataset['b_var'][0][0]
    # m/s
    robot_trans_speed = dataset['v']
    robot_trans_speed.shape = (12609)
    # m^2/s^2
    robot_trans_speed_var = dataset['v_var'][0][0]
    # rad/s
    robot_rot_speed = dataset['om']
    robot_rot_speed.shape = (12609)
    # rad^2/s^2
    robot_rot_speed_var = dataset['om_var'][0][0]
    # the distance, d, between the center of the robot and the laser rangefinder [m]
    d  = dataset['d'][0][0]

    return timestamps, robot_true_x, robot_true_y, robot_true_th, landmark_true_pos, landmark_estimated_range, landmark_estimated_range_var, landmark_estimated_bearing, landmark_estimated_bearing_var, robot_trans_speed, robot_trans_speed_var, robot_rot_speed, robot_rot_speed_var, d 

def calc_new_position(current_x, current_y, current_th, trans_speed, rot_speed, time) :
    """ Calculates new position of robot using motion model given :
        current_x : _ meters 
        current_y : _ meters
        current_th : _ radlandmark_estimated_range
        trans_speed : _ meters/second 
        rot_speed : _ rad/second 
        time : _ seconds
    """
    
    pos_prev = np.array([current_x, current_y, current_th])
    ang_mat = np.array([[np.cos(current_th),0],[np.sin(current_th), 0], [0,1]])
    control = np.array([trans_speed, rot_speed])
    new_pos =  pos_prev + time * ang_mat @ control
    return new_pos
    
def ret_sensor_jacobian_l(x_t, r_l, d):
    """
    calculates jacobian for lth landmark
    at timestep t
    Arguments: d, current belief, and landmark position
    return 2*3 jacobian
    """
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
    j_12 = d * (a * np.cos(theta_k) + b*np.sin(theta_k))/(t*t) 
    jacob = np.array([[j_00,j_01,j_02],[j_10,j_11,j_12]])
    return jacob


def observation_model(pos, landmark_true_pos, d):
    """
    return 34*1
    take landmark_true_pos as all the landmarks
    """
    output = np.zeros((34,1))
    for i in range(landmark_true_pos.shape[0]) :
        output[2*i] = np.sqrt((landmark_true_pos[i,0]-pos[0]-d*np.cos(pos[2]))**2 + (landmark_true_pos[i,1]-pos[1]-d*np.sin(pos[2]))**2)
        output[2*i+1] = atan2(landmark_true_pos[i,1]-pos[1]-d*np.sin(pos[2]),landmark_true_pos[i,0]-pos[0]-d*np.cos(pos[2])) - pos[2]
    
    return output

def ret_sensor_jacobian(x_t, landmark_estimated_range_t, landmark_estimated_bearing_t, d):
    for i in range()



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


# !!!!!!!! Aryan read below
# function g  == calc_new_pos ? Afaik it's this, you can just sub existing function here.
# function h we have to write 
    
def h(pos, r_l):
    """
    return 34*1
    take r_l as all the landmarks(17*2)
    """
    return




def ekf(prev_pos, prev_cov, control, obs, d, Rt, Qt, landmarks):
    """ Takes prev position, prev_cov, current control ,and current obs """
    G = ret_motion_jacobian(prev_th=prev_pos[2],trans_speed=control[0])
    H = ret_sensor_jacobian(d)

    pos_current_dash = calc_new_position(prev_pos[0], prev_pos[1], prev_pos[2], control[0], control[1], time)

    # pos_current_dash = calc_new_pos(current_control,prev_pos)
    current_cov_dash = G@prev_cov@G.T + Rt
    
    kalman_gain = current_cov_dash@H.T@np.linalg.inv(H@current_cov_dash@H.T + Qt)
    pos_current = pos_current_dash + kalman_gain@(obs - h(pos_current_dash))
    cov_current = (np.eye(3) - kalman_gain@H)@current_cov_dash
    
    return pos_current, cov_current
# main 



timestamps, robot_true_x, robot_true_y, robot_true_th, landmark_true_pos, landmark_estimated_range, landmark_estimated_range_var, landmark_estimated_bearing, landmark_estimated_bearing_var, robot_trans_speed, robot_trans_speed_var, robot_rot_speed, robot_rot_speed_var, d = read_data('dataset.npz')
number_of_steps = 2
Rt = np.diag([robot_trans_speed_var,robot_trans_speed_var, robot_rot_speed_var])
Qt = np.zeros((34,34))
for i in range(34):
    if i%2==0:
        Qt[i][i] = landmark_estimated_range_var
    else:
        Qt[i][i] = landmark_estimated_bearing_var

#fig = plt.figure(figsize=(10,5))
#plt.scatter(landmark_true_pos[:,0], landmark_true_pos[:,1], color = 'r', marker='x')
#plt.plot(robot_true_x[0:number_of_steps], robot_true_y[0:number_of_steps])

new_pos = []
new_pos.append([robot_true_x[0], robot_true_y[0], robot_true_th[0]])
for i in range(1,number_of_steps):
    temp = calc_new_position(new_pos[i-1][0],new_pos[i-1][1],new_pos[i-1][2],robot_trans_speed[i-1], robot_rot_speed[i-1],0.1)
    jac = ret_motion_jacobian(new_pos[i-1][2],robot_trans_speed[i-1])
    print(jac)
    new_pos.append(temp)

new_pos = np.array(new_pos)
#plt.plot(new_pos[:,0], new_pos[:,1], 'r')