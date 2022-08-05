import numpy as np
from simulator.visualizer import DataLogger
from simulator.timeProfiling import timeProfiling
from control_lib.goToGoal import Pcontrol
from control_lib.safetyCBF import safetyCBF
import pickle

import rospy
from geometry_msgs.msg import Pose2D,Twist
import signal

# Constants and Settings
save_data = True # log data using pickle
defname = 'experiment_result/ROSTB_FormationObstacle'
if save_data: fdata_vis = defname + '_data.pkl'


# Go to Goal Setup
# ------------------------------------------------------
cform_goal = np.array([1.4,-0.2, 0])
form_w = 1 # width of rectangle formation
form_h = 0.5 # height of rectangle formation
# Set desired formation position 
# --> Order: red, blue, green, orange
# --> rotated -90deg in final configuration (from the setup in ROSTB_GoToGoal)
goal_pos = np.array([   cform_goal + np.array([ -form_h/2, -form_w/2, 0]), 
                        cform_goal + np.array([  form_h/2, -form_w/2, 0]), 
                        cform_goal + np.array([  form_h/2,  form_w/2, 0]), 
                        cform_goal + np.array([ -form_h/2,  form_w/2, 0]),  ])

# CBF SETUP
# ------------------------------------------------------
USECBF_BOUNDARY = False
# Set field size (based on camera view)
scale_m2pxl = 363.33
half_width = 1920 / (2 * scale_m2pxl)
half_height = 1080 / (2 * scale_m2pxl)
field_x = [-half_width, half_width]
field_y = [-half_height, half_height]

USECBF_STATICOBS = True
# Define Obstacle location
obstacle = []
obstacle += [{"pos": np.array([-0.4, 0.6, 0]), "r": 0.3}]
obstacle += [{"pos": np.array([-0.5, -1., 0]), "r": 0.3}]
gamma_staticObs = 1000

USECBF_FORMATION = True
form_d =np.sqrt(np.power(form_w,2) + np.power(form_h,2)) # Diagonal of rectangle
form_A = np.array([ [      0, form_h, form_d, form_w],
                    [ form_h,      0, form_w, form_d],
                    [ form_d, form_w,      0, form_h],
                    [ form_w, form_d, form_h, 0     ] ])
form_epsilon = 0.05 # tolerance for maintaining distance
gamma_form = 1000

# define sensing range (to ease optimization)
sr = 10 # in meter


# ROS & Turtlebots specific parameters
# ------------------------------------------------------
ROS_RATE = 50
robot_names =['tb3_0','tb3_1','tb3_2','tb3_3']
lookAhead_l = 0.06
global_poses = [None]*len(robot_names)
# NOTES: it seems cleaner to do it this way 
# rather than dynamically creating the callbacks
def pos_callback_0( data ): global_poses[0] = data
def pos_callback_1( data ): global_poses[1] = data
def pos_callback_2( data ): global_poses[2] = data
def pos_callback_3( data ): global_poses[3] = data

def si_to_TBTwist(u, theta):
    # Inverse Look up ahead Mapping (u_z remain 0.)
    #   V = u_x cos(theta) + u_y sin(theta)
    #   omg = (- u_x sin(theta) + u_y cos(theta)) / l
    vel_lin = u[0]*np.cos(theta) + u[1]*np.sin(theta)
    vel_ang = (- u[0]*np.sin(theta) + u[1]*np.cos(theta))/lookAhead_l

    # TODO: do max (or saturation) if needed
    TBvel = Twist()
    TBvel.linear.x = vel_lin
    TBvel.linear.y = 0
    TBvel.linear.z = 0
    TBvel.angular.x = 0
    TBvel.angular.y = 0
    TBvel.angular.z = vel_ang

    return TBvel


class Computation():
    def __init__(self):

        self.robot_num = len(robot_names)

        # INITIALIZE ROS SUBSCRIBER and Publisher
        rospy.Subscriber('/{}/pos'.format(robot_names[0]), Pose2D, pos_callback_0)
        rospy.Subscriber('/{}/pos'.format(robot_names[1]), Pose2D, pos_callback_1)
        rospy.Subscriber('/{}/pos'.format(robot_names[2]), Pose2D, pos_callback_2)
        rospy.Subscriber('/{}/pos'.format(robot_names[3]), Pose2D, pos_callback_3)
        self.ros_pubs = []
        for i in range(self.robot_num): 
            self.ros_pubs += [rospy.Publisher('/{}/cmd_vel'.format(robot_names[i]),Twist, queue_size=1)]


        # TIME PROFILING
        self.ctrlProfiling = timeProfiling('Controller')
        
        # INITIALIZE CONTROLLER and DATA LOGGER
        self.Pgain = 0.8 # for go-to-goal

        # Simulate communication with Laplacian
        # Each robot can get the other robot info, but only process within sensing region
        self.inNeigh = [np.where(form_A[:,i] > 0)[0] for i in range(self.robot_num)]

        self.cbf = [safetyCBF( sensingRange=sr ) for _ in range(self.robot_num)]
        self.data_log = [DataLogger() for _ in range(self.robot_num)]
        # Register the static obstacle to the controller
        for i in range(self.robot_num):
            if USECBF_BOUNDARY:
                self.cbf[i].registerBound( 0, [min(field_x)-0.1, max(field_x)+0.1]) # X boundary
                self.cbf[i].registerBound( 1, [min(field_y)-0.1, max(field_y)+0.1]) # Y boundary
                self.data_log[i].data_register('h_bound', np.zeros(3)) 

            if USECBF_FORMATION:
            # register other robots (in neighbors Laplacian) to maintain Formation
                for j in self.inNeigh[i]: self.cbf[i].registerFormationLink( j, form_A[i,j], form_epsilon, gamma=gamma_form, power=3) 
                self.data_log[i].data_register('h_formation', np.zeros(2*len(self.inNeigh[i]))) 

            if USECBF_STATICOBS:            
                for k in range(len(obstacle)):
                    self.cbf[i].registerStaticObs(obstacle[k]["pos"], obstacle[k]["r"], gamma=gamma_staticObs, power=3) # Static Obstacle
                self.data_log[i].data_register('h_staticObs', np.zeros(len(obstacle))) 

            # Assign maximum allowed directional velocity
            # self.cbf[i].register_velocity_bound( 0.12 )
            # self.cbf[i].register_unicycle_vel_bound( 0.15, 2., lookAhead_l)
        
        # Add handler if CTRL+C is pressed --> then save data to pickle
        signal.signal(signal.SIGINT, self.signal_handler)


    # Allow CTRL+C to stop the controller and dump the log into pickle
    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C. Turning off the controller.')

        # LOG the data from simulation
        if save_data:
            print('Storing the data to files...', flush=True)
            with open(fdata_vis, 'wb') as f:
                pickle.dump(dict(data_log=self.data_log), f)
            print('Done.')
        exit() # Force Exit


    # MAIN LOOP CONTROLLER & VISUALIZATION
    def loop_sequence(self, it = 0):
        # Showing Time Stamp
        if (it > 0) and (it % ROS_RATE == 0):
            t = it/ROS_RATE
            print('Experiment t = {}s.'.format(t))
            self.ctrlProfiling.printStatus()

        if all (v is not None for v in global_poses):
            # Only run if all position values are initialized by localization systems
            
            for i in range(self.robot_num):
                # Collect States
                # ------------------------------------------------
                # TODO: checking if the pose is not None
                current_q = np.array([global_poses[i].x, global_poses[i].y, 0]) # Get position data only
                goal = goal_pos[i]

                if USECBF_FORMATION:
                    # Simulate get other robots position with laplacian
                    for j in self.inNeigh[i]: # TODO: sensing range?
                        j_pos = np.array([global_poses[j].x, global_poses[j].y, 0])
                        self.cbf[i].updateFormationLink( j, j_pos)

                # update theta
                self.cbf[i].update_current_theta(global_poses[i].theta)


                # Profiling for controller computation time
                self.ctrlProfiling.startTimer()
                #################################################

                # Implementation of Control
                # ------------------------------------------------
                # Calculate nominal controller
                u_nom = Pcontrol(current_q, self.Pgain, goal)

                # set speed limit
                speed_limit = 0.1
                norm = np.hypot(u_nom[0], u_nom[1])
                if norm > speed_limit: u_nom = speed_limit* u_nom / norm # max 

                # Ensure safety (obstacle avoidance)
                u, h = self.cbf[i].computeSafeController(current_q, u_nom)

                # Send command # TODO
                # ------------------------------------------------
                self.ros_pubs[i].publish( si_to_TBTwist(u, global_poses[i].theta) )

                #################################################
                self.ctrlProfiling.stopTimer() # Stop & Process profiling data

                ## Additional Logging Variables
                if USECBF_BOUNDARY : self.data_log[i].data_store_temp('h_bound', h['h_bound'])
                if USECBF_STATICOBS: self.data_log[i].data_store_temp('h_staticObs', h['h_staticObs'])
                if USECBF_FORMATION: self.data_log[i].data_store_temp('h_formation', h['h_formation'])
                ## The log is stored when update_history is called (inside update_simulation)
                self.data_log[i].update_history({'q':current_q}, {'q':u}, {'u':u}, 1/ROS_RATE)


def main():
    comp = Computation()
    try:
        rospy.init_node('consesus_cntroller', anonymous=True)
        #current_time = rospy.Time.now()
        #last_time = rospy.Time.now()
        r = rospy.Rate(ROS_RATE)
        
        it = 0
        while not rospy.is_shutdown():

            comp.loop_sequence(it)
            it +=1
            r.sleep()

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()