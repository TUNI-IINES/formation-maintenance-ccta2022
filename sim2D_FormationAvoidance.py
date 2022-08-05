import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator.visualizer import Visualizer2D
from simulator.timeProfiling import timeProfiling
from control_lib.goToGoal import Pcontrol
from control_lib.safetyCBF import safetyCBF
import pickle


#mpl.rcParams['animation.ffmpeg_path'] = r'/home/zephyr/anaconda3/bin/ffmpeg'

# Constants and Settings
Ts = 0.1 # in second. Determine Visualization and dynamic update speed
tmax = 30 # simulation duration in seconds (only works when save_animate = True)
save_animate = False # True: saving but not showing, False: showing animation but not real time
showCleanVer = False # Clean Version of visualization
save_data = True # log data using pickle
use_unicycle = True
lookAhead_l = 0.06

defname = 'animation_result/sim2D_FormationAvoidance'
if showCleanVer:
    fname_output = r''+defname+'_clean.gif'
    trajTail = int(4/Ts) # Only show past 4 second trajectory
else:
    fname_output = r''+defname+'.gif'
    trajTail = None # Show all trajectory
if save_data: fdata_vis = defname + '_vis.pkl'

# Define Two formations
cform1_init = np.array([-1., 1., 0])
cform2_init = np.array([-1.02, -1, 0])
cform1_goal = np.array([1., -1., 0])
cform2_goal = np.array([1., 1., 0])

USECBF_BOUNDARY = True
USECBF_FORMATION = True
USECBF_ELLIPSEAV = True

form_l = 0.5 # length between two robots in a formation
form_epsilon = 0.05 # tolerance for maintaining distance
major_l = 2*0.6
minor_l = 2*0.4
gamma_ellipsAv = 0.2

# Set initial formation position --> Order: red, blue, green, orange
init_pos = np.array([   cform1_init + np.array([ 0,  form_l/2, 0]), 
                        cform1_init + np.array([ 0, -form_l/2, 0]), 
                        cform2_init + np.array([ 0,  form_l/2, 0]), 
                        cform2_init + np.array([ 0, -form_l/2, 0]),  ])
# Set desired formation position --> rotated -90deg in final configuration
goal_pos = np.array([   cform1_goal + np.array([  form_l/2, 0, 0]), 
                        cform1_goal + np.array([ -form_l/2, 0, 0]), 
                        cform2_goal + np.array([ -form_l/2, 0, 0]), 
                        cform2_goal + np.array([  form_l/2, 0, 0]),  ])

form_A = np.array([ [      0, form_l,      0,      0],
                    [ form_l,      0,      0,      0],
                    [      0,      0,      0, form_l],
                    [      0,      0, form_l, 0     ] ])
form_id = [0, 0, 1, 1] # Identifier of each group
gamma_form = 2000

# Set field size for plotting (based on camera view)
scale_m2pxl = 363.33
half_width = 1920 / (2 * scale_m2pxl)
half_height = 1080 / (2 * scale_m2pxl)
field_x = [-half_width, half_width]
field_y = [-half_height, half_height]

# define sensing range (to ease optimization)
sr = 10 # in meter # It needs to be greater than major length
ds_dyn = 0.1

def compute_ellipse_form2(leader_pos, other_pos):
    # Compute ellipse centroids
    cent = (leader_pos + other_pos)/2
     # Compute vector for each ellipse's theta
    vector = leader_pos - cent # robot 0 is leader group 1
    theta = np.arctan2(vector[1], vector[0]) # no offset for now
    return {'cent':cent, 'theta': theta}

class Simulate():
    def __init__(self, visualizer):
        self.vis = visualizer
        # TIME PROFILING
        self.ctrlProfiling = timeProfiling('Controller')
        self.visProfiling = timeProfiling('Visualizer')

        # SCENARIO SETTINGS
        if use_unicycle:
            self.vis.spawn_robots(init_pos, model_name='Unicycle', Ts=Ts) # Spawn N unicycle dynamics
        else:
            self.vis.spawn_single_integrator(init_pos, Ts=Ts) # Spawn N single integrator dynamics
        self.robot_num = self.vis.getRobotNum()

        # INITIALIZE CONTROLLER
        self.Pgain = 0.8 # for go-to-goal

        # Simulate communication with Laplacian
        # Each robot can get the other robot info, but only process within sensing region
        self.inNeigh = [np.where(form_A[:,i] > 0)[0] for i in range(self.robot_num)]
        # TODO: maybe it is better if sensing range is implemented in here separately from CBF

        self.cbf = [safetyCBF( sensingRange=sr ) for _ in range(self.robot_num)]
        # Register the static obstacle to the controller
        for i in range(self.robot_num):
            if USECBF_BOUNDARY:
                self.cbf[i].registerBound( 0, [min(field_x)-0.1, max(field_x)+0.1]) # X boundary
                self.cbf[i].registerBound( 1, [min(field_y)-0.1, max(field_y)+0.1]) # Y boundary
                self.vis.data_log[i].data_register('h_bound', np.zeros(3)) 

            # register one ellipse obstacle, representing the other group
            if USECBF_ELLIPSEAV:
                id = 1 if (form_id[i] == 0) else 0
                self.cbf[i].registerEllipseObs( id, major_l, minor_l, gamma=gamma_ellipsAv, power=3)
                self.vis.data_log[i].data_register('h_ellipseObs', np.zeros(1))

            # register other robots (in neighbors Laplacian) to maintain Formation
            if USECBF_FORMATION:
                for j in self.inNeigh[i]: self.cbf[i].registerFormationLink( j, form_A[i,j], form_epsilon, gamma=gamma_form, power=3)  
                self.vis.data_log[i].data_register('h_formation', np.zeros(2*len(self.inNeigh[i]))) 

            # Assign maximum allowed directional velocity
            self.cbf[i].register_velocity_bound( 0.15 )
            # self.cbf[i].register_unicycle_vel_bound( 0.2, 2.5, lookAhead_l)
            self.vis.set_lookUpAhead_param(i, lookAhead_l)

        # INITIALIZE PLOTTING
        # Register additional plot >> for now it is strongly coupled with log
        if use_unicycle: self.vis.setPlot_withIcon('unicycle')
        if showCleanVer: 
            self.vis.setPlotOnly2D()
        else:
            # Additional plot data should live in log
            if USECBF_BOUNDARY: self.vis.registerAdditionalPlot('h_bound')
            if USECBF_FORMATION: self.vis.registerAdditionalPlot('h_formation')
            if USECBF_ELLIPSEAV: self.vis.registerAdditionalPlot('h_ellipseObs')
        # Register Additional Plot should be called before plot2DTraj
        self.vis.plot2DTraj()
        # Add goal points
        for i in range(len(goal_pos)):
            self.vis.drawCircle( (goal_pos[i][0], goal_pos[i][1]), 0.03, col='g')
        # Store values for the plottings
        self.ellipse_form = [None]*2
        comp_pos = { i:init_pos[i] for i in range(self.robot_num)}
        if use_unicycle: 
            for i in range(self.robot_num):
                th = 0
                comp_pos[i] = comp_pos[i] + lookAhead_l*np.array([np.cos(th), np.sin(th), 0])
        self.ellipse_form[0] = compute_ellipse_form2(comp_pos[0],  comp_pos[1])
        self.ellipse_form[1] = compute_ellipse_form2(comp_pos[2],  comp_pos[3])
        # Add ellipse formation
        for id in range(2):
            self.vis.drawMovingEllipseFormation(id, self.ellipse_form[0]['cent'], major_l, minor_l)
        
        # Limit visualization on field area
        self.vis.setPlotField(0, field_x)
        self.vis.setPlotField(1, field_y)


    # MAIN LOOP CONTROLLER & VISUALIZATION
    def loop_sequence(self, i = 0):
        # Showing Time Stamp
        if (i > 0) and (i % round(1/Ts) == 0):
            t = round(i*Ts)
            print('simulating t = {}s.'.format(t))
            self.ctrlProfiling.printStatus()
            self.visProfiling.printStatus()

        for i in range(self.robot_num):
            # Collect States
            # ------------------------------------------------
            current_state = self.vis.getRobotState(i)
            current_q = current_state["q"] # Get position data only
            if use_unicycle: 
                current_theta = current_state["theta"]
                ell_si = lookAhead_l*np.array([np.cos(current_theta), np.sin(current_theta), 0])
                current_q = current_q + ell_si # Get position data at point ahead
                # update theta
                self.cbf[i].update_current_theta(current_theta)
            goal = goal_pos[i]

            if USECBF_FORMATION:
                # Simulate get other robots position with laplacian
                for j in self.inNeigh[i]: # TODO: sensing range?
                    #self.cbf[i].updateDynamicObs( j, self.vis.getRobotState(j)["q"], Ts)
                    j_q = self.vis.getRobotState(j)["q"]
                    if use_unicycle: 
                        j_theta = self.vis.getRobotState(j)["theta"]
                        ell_sj = lookAhead_l*np.array([np.cos(j_theta), np.sin(j_theta), 0])
                        j_q = j_q + ell_sj # Get position data at point ahead
                    self.cbf[i].updateFormationLink( j, j_q)

            # Simulate communication between form.leader and distribute to all agent
            # TODO: properly do this
            if USECBF_ELLIPSEAV: 
                id = 1 if (form_id[i] == 0) else 0
                self.cbf[i].updateEllipseObs(id, self.ellipse_form[id]['cent'], self.ellipse_form[id]['theta'])

            # Profiling for controller computation time
            self.ctrlProfiling.startTimer()
            #################################################

            # Implementation of Control
            # ------------------------------------------------
            # Calculate nominal controller
            u_nom = Pcontrol(current_q, self.Pgain, goal)

            # # set speed limit
            # speed_limit = 0.2
            # norm = np.hypot(u_nom[0], u_nom[1])
            # if norm > speed_limit: u_nom = speed_limit* u_nom / norm # max 

            # Ensure safety (obstacle avoidance)
            u, h = self.cbf[i].computeSafeController(current_q, u_nom)

            # Send command
            # ------------------------------------------------
            self.vis.sendRobotCommand(i, u)

            #################################################
            self.ctrlProfiling.stopTimer() # Stop & Process profiling data

            # Additional Logging Variables
            if USECBF_BOUNDARY : self.vis.data_log[i].data_store_temp('h_bound', h['h_bound'])
            if USECBF_FORMATION: self.vis.data_log[i].data_store_temp('h_formation', h['h_formation'])
            if USECBF_ELLIPSEAV: self.vis.data_log[i].data_store_temp('h_ellipseObs', h['h_ellipseObs'])
            # The log is stored when update_history is called (inside update_simulation)

        self.visProfiling.startTimer()
        #################################################
        # Update drawing
        self.vis.update_simulation()

        # Simulate getting other group's computation via communication
        # TODO: implement this properly, for now this is just proof of concept
        temp_state = {i:self.vis.getRobotState(i) for i in range(self.robot_num)}
        comp_pos = { i:temp_state[i]["q"] for i in range(self.robot_num)}
        if use_unicycle: 
            for i in range(self.robot_num):
                th = temp_state[i]["theta"]
                comp_pos[i] = comp_pos[i] + lookAhead_l*np.array([np.cos(th), np.sin(th), 0])
        self.ellipse_form[0] = compute_ellipse_form2(comp_pos[0],  comp_pos[1])
        self.ellipse_form[1] = compute_ellipse_form2(comp_pos[2],  comp_pos[3])

        # TODO: alternatively implement formation object, 
        #       but this might make the ROS implementation harder
        for id in range(2):
            self.vis.updateMovingEllipseFormation(id, self.ellipse_form[id]['cent'], self.ellipse_form[id]['theta'])

        self.vis.update_plot2DTraj(shortTail=trajTail)


        #################################################
        self.visProfiling.stopTimer() # Stop & Process profiling data


def main():
    print("start")
    simulationTime = timeProfiling('Total Simulation')
    simulationTime.startTimer() # start timer
    #################################################

    # Ìnitialize Visualizer
    vis = Visualizer2D()
    # Step through simulation
    # ------------------------------------------------
    sim_iter = round(tmax/Ts)
    sim = Simulate(vis)
    ani = animation.FuncAnimation(vis.fig, sim.loop_sequence, save_count=sim_iter, interval = Ts*1000)
    if save_animate: # default not showing animation
        print('saving animation ...')
        ani.save(fname_output, fps=round(1/Ts))
        #ani.save(r'animation.mp4', writer=animation.FFMpegWriter(fps=round(1/dt)))
    else:
        plt.show()

    # NOTE: below code only run when save_animate=True and once finished saving
    #################################################
    simulationTime.stopShowElapsed() # Stop & show elapsed time

    # LOG the data from simulation
    if save_data:
        print('Storing the data to files...', flush=True)
        with open(fdata_vis, 'wb') as f:
            pickle.dump(dict(data_log=sim.vis.data_log), f)
        
        print('Done.')

if __name__ == '__main__':
    main()