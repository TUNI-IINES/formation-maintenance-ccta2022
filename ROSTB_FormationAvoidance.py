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
defname = 'experiment_result/ROSTB_FormationAvoidance'
if save_data: fdata_vis = defname + '_data.pkl'


# Go to Goal Setup
# ------------------------------------------------------
cform1_goal = np.array([1., -1., 0])
cform2_goal = np.array([1., 1., 0])
#cform2_goal = np.array([-1.6, -0, 0])
form_l = 0.5 # length between two robots in a formation
# Set desired formation position 
# --> Order: red, blue, green, orange
goal_pos = np.array([   cform1_goal + np.array([  form_l/2, 0, 0]), 
                        cform1_goal + np.array([ -form_l/2, 0, 0]), 
                        cform2_goal + np.array([ -form_l/2, 0, 0]), 
                        cform2_goal + np.array([  form_l/2, 0, 0]),  ])
# CBF SETUP
# ------------------------------------------------------
USECBF_BOUNDARY = False
# Set field size (based on camera view)
scale_m2pxl = 363.33
half_width = 1920 / (2 * scale_m2pxl)
half_height = 1080 / (2 * scale_m2pxl)
field_x = [-half_width, half_width]
field_y = [-half_height, half_height]

USECBF_FORMATION = True
form_A = np.array([ [      0, form_l,      0,      0],
                    [ form_l,      0,      0,      0],
                    [      0,      0,      0, form_l],
                    [      0,      0, form_l, 0     ] ])
form_epsilon = 0.05 # tolerance for maintaining distance
form_id = [0, 0, 1, 1] # Identifier of each group
gamma_form = 1000

USECBF_ELLIPSEAV = True
major_l = 2*0.6
minor_l = 2*0.4

def compute_ellipse_form(leader_pos, other_pos):
    # Compute ellipse centroids
    cent = (leader_pos + other_pos)/2
     # Compute vector for each ellipse's theta
    vector = leader_pos - cent # robot 0 is leader group 1
    theta = np.arctan2(vector[1], vector[0]) # no offset for now
    return {'cent':cent, 'theta': theta}
gamma_ellipsAv = 0.1

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

            # register one ellipse obstacle, representing the other group
            if USECBF_ELLIPSEAV:
                id = 1 if (form_id[i] == 0) else 0
                self.cbf[i].registerEllipseObs( id, major_l, minor_l, gamma=gamma_ellipsAv, power=3)
                self.data_log[i].data_register('h_ellipseObs', np.zeros(1))

            # Assign maximum allowed directional velocity
            # self.cbf[i].register_velocity_bound( 0.15 )
            # self.cbf[i].register_unicycle_vel_bound( 0.2, 2.5, lookAhead_l)
        
        self.ellipse_form = [None]*2
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

            # Simulate getting other group's computation via communication
            # TODO: implement this properly, for now this is just proof of concept
            self.ellipse_form[0] = compute_ellipse_form(
                np.array([global_poses[0].x, global_poses[0].y, 0]),
                np.array([global_poses[1].x, global_poses[1].y, 0]))
            self.ellipse_form[1] = compute_ellipse_form(
                np.array([global_poses[2].x, global_poses[2].y, 0]),
                np.array([global_poses[3].x, global_poses[3].y, 0]))


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

                # Simulate communication between form.leader and distribute to all agent
                # TODO: properly do this
                if USECBF_ELLIPSEAV: 
                    id = 1 if (form_id[i] == 0) else 0
                    self.cbf[i].updateEllipseObs(id, self.ellipse_form[id]['cent'], self.ellipse_form[id]['theta'])

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
                if USECBF_FORMATION: self.data_log[i].data_store_temp('h_formation', h['h_formation'])
                if USECBF_ELLIPSEAV: self.data_log[i].data_store_temp('h_ellipseObs', h['h_ellipseObs'])
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