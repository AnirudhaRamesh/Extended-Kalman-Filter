import numpy as np
import matplotlib.pyplot as plt

def read_data(filename) :
    """ Function to read data from given input dataset with the format of "Lost in the woods" dataset. """
    dataset = np.load(filename)
    # secs
    timestamps = dataset['t']
    # meters
    robot_true_x = dataset['x_true']
    robot_true_y = dataset['y_true']
    robot_true_th = dataset['th_true']
    landmark_true_pos = dataset['l']
    landmark_estimated_range = dataset['r']
    # meter^2
    landmark_estimated_range_var = dataset['r_var']
    # rad
    landmark_estimated_bearing = dataset['b']
    # rad^2
    landmark_estimated_bearing_var = dataset['b_var']
    # m/s
    robot_trans_speed = dataset['v']
    # m^2/s^2
    robot_trans_speed_var = dataset['v_var']
    # rad/s
    robot_rot_speed = dataset['om']
    # rad^2/s^2
    robot_rot_speed_var = dataset['om_var']
    # the distance, d, between the center of the robot and the laser rangefinder [m]
    d  = dataset['d']

    return timestamps, robot_true_x, robot_true_y, robot_true_th, landmark_true_pos, landmark_estimated_range, landmark_estimated_range_var, landmark_estimated_bearing, landmark_estimated_bearing_var, robot_trans_speed, robot_trans_speed_var, robot_rot_speed, robot_rot_speed_var, d 

def calc_new_position(current_x, current_y, current_th, trans_speed, rot_speed, time) :
    """ Calculates new position of robot using motion model given :
        current_x : _ meters 
        current_y : _ meters
        current_th : _ rad
        trans_speed : _ meters/second 
        rot_speed : _ rad/second 
        time : _ seconds
    """
    
    pos_prev = np.array([current_x, current_y, current_th])
    ang_mat = np.array([[np.cos(current_th)[0],0],[np.sin(current_th)[0], 0], [0,1]])
    control = np.array([trans_speed, rot_speed])
    new_pos =  pos_prev + time * ang_mat @ control
    return new_pos
    
def ret_sensor_jacobian():
	return

def ret_motion_jacobian(u_t, x_t_1, var):
	"""
	Takes control, previous position, and variance
	change accordingly
	"""
	return

# main 
timestamps, robot_true_x, robot_true_y, robot_true_th, landmark_true_pos, landmark_estimated_range, landmark_estimated_range_var, landmark_estimated_bearing, landmark_estimated_bearing_var, robot_trans_speed, robot_trans_speed_var, robot_rot_speed, robot_rot_speed_var, d = read_data('dataset.npz')
number_of_steps = 12609 
fig = plt.figure(figsize=(10,5))
plt.scatter(landmark_true_pos[:,0], landmark_true_pos[:,1], color = 'r', marker='x')
plt.plot(robot_true_x[0:number_of_steps], robot_true_y[0:number_of_steps])

new_pos = []
new_pos.append([robot_true_x[0], robot_true_y[0], robot_true_th[0]])
for i in range(1,number_of_steps):
    temp = calc_new_position(new_pos[i-1][0],new_pos[i-1][1],new_pos[i-1][2],robot_trans_speed[i-1], robot_rot_speed[i-1],0.1)
    new_pos.append(temp)

new_pos = np.array(new_pos)
plt.plot(new_pos[:,0], new_pos[:,1], 'r')