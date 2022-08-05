import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from simulator.dynamics import SingleIntegrator, Unicycle, SI_DroneVision

class DataLogger():
    # Keeps track of state and input history for plotting.
    def __init__(self):
        self.time = []
        self.state = {}
        self.input = {}
        self.dot_state = {}
            
        self.min_state = {}
        self.max_state = {}
        self.min_input = {}
        self.max_input = {}

        # accomodate for additional data
        self.data_temp = {}
        self.data = {}

        self.min_data = {}
        self.max_data = {}

    def update_history(self, new_state, new_dot_state, new_input, dt):
        # Appends current state and input for plotting.
        if not self.time: # Empty List. first time called
            t = 0 # initialize time
            # Initialize each state and dot_state with empty array based on each state's dimension
            # dot_state and state should have the same keys
            for key, val in new_state.items():
                # TODO: consider max number of value to store (based on simulation time)
                # val.size is the dimension of the state[key], e.g., xyz pos is 3
                if np.isscalar(val): val = np.array([val])
                self.state[key] = np.empty(shape=[0, val.size])
                self.dot_state[key] = np.empty(shape=[0, val.size])
                # Keep track of min and max (at the moment, no need to track dot_state)
                self.min_state[key] = np.copy(val).astype(float)
                self.max_state[key] = np.copy(val).astype(float)
            
            # Repeat with initialize each input
            for key, val in new_input.items():
                # TODO: consider max number of value to store (based on simulation time)
                # val.size is the dimension of the input[key], e.g., xyz pos is 3
                if np.isscalar(val): val = np.array([val])
                self.input[key] = np.empty(shape=[0, val.size])
                # Keep track of min and max (at the moment, no need to track dot_state)
                self.min_input[key] = np.copy(val).astype(float)
                self.max_input[key] = np.copy(val).astype(float)

        else: # TODO. check if I can store them together in here
            t = self.time[-1] + dt
            # Update min and max data for state
            for key, val in new_state.items():
                if np.isscalar(val): val = np.array([val])
                for i in range(val.size):
                    if val[i] < self.min_state[key][i]: self.min_state[key][i] = val[i] 
                    if val[i] > self.max_state[key][i]: self.max_state[key][i] = val[i] 
            # Update min and max data for input
            for key, val in new_input.items():
                if np.isscalar(val): val = np.array([val])
                for i in range(val.size):
                    if val[i] < self.min_input[key][i]: self.min_input[key][i] = val[i] 
                    if val[i] > self.max_input[key][i]: self.max_input[key][i] = val[i] 
            # Update min and max data for log_data
            for key, val in self.data_temp.items():
                if np.isscalar(val): val = np.array([val])
                for i in range(val.size):
                    if val[i] < self.min_data[key][i]: self.min_data[key][i] = val[i]
                    if val[i] > self.max_data[key][i]: self.max_data[key][i] = val[i]

        # TODO: if we set max array then assign it instead of append
        # Store the state and input
        for key, val in new_state.items(): 
            dot_val = new_dot_state[key]
            # Exception for scalar values
            if np.isscalar(val): 
                val = np.array([val])
                dot_val = np.array([dot_val])
            self.state[key] = np.append(self.state[key], np.array([val]), axis=0)
            self.dot_state[key] = np.append(self.dot_state[key], np.array([dot_val]), axis=0)
        for key, val in new_input.items(): 
            if np.isscalar(val): val = np.array([val])
            self.input[key] = np.append(self.input[key], np.array([val]), axis=0)
        self.time.append(t)
        # store additional log from log_temp to log_data
        for key, val in self.data_temp.items(): 
            self.data[key] = np.append(self.data[key], np.array([val]), axis=0)

    # ACCOMODATE ADDITIONAL DATA TO BE LOGGED
    # Given the rule that:
    # (1) The data to be added need to be known beforehand, all initialized together before new data is added
    # (2) The update has to be sync with the update of the main simulation data (state and input).
    #     This is done by finishing log_store_temp before calling update_history.
    #     --> if log_store_temp is not called in a given time-step, 
    #           it will use the last assigned value
    #     --> if log_store_temp is called multiple time, 
    #           then it will only store the last one when update_history is called

    def data_register(self, name, init_val): 
        assert name not in self.data, \
            f"log_register >> assigned key {name} already exist. \
            Please register another key."

        # TODO: if it is multidimensional np array, consider flattening
        #   and saving the dimension so it can be reconstructed again later.
        if not isinstance(init_val, np.ndarray): 
            init_val = np.array([init_val]) # --> should be 1D np array

        # Flatten the input if it is not 1D array
        self.data_temp[name] = init_val if init_val.ndim == 1 else init_val.flatten()
        self.data[name] = np.empty(shape=[0, init_val.size])
        # Exceptionally, the first append is done in here, 
        # because spawn robot (including update_history) is done before data_register
        self.data[name] = np.append(self.data[name], np.array([init_val]), axis=0)

        self.min_data[name] = np.copy(init_val).astype(float)
        self.max_data[name] = np.copy(init_val).astype(float)        


    def data_store_temp(self, name, new_val): 
        assert name in self.data_temp, \
            f"log_store_temp >> assigned key {name} not yet exist. \
            Make sure you already call log_register"

        # TODO: if it is multidimensional np array, consider flattening
        #   and saving the dimension so it can be reconstructed again later.
        if not isinstance(new_val, np.ndarray): 
            new_val = np.array([new_val]) # --> should be 1D np array

        # Flatten the input if it is not 1D array
        self.data_temp[name] = new_val if new_val.ndim == 1 else new_val.flatten()
        # The append will be done when update_history is called at the end

    # get min and max for state and input (for any xyz - idx)
    def get_min_state(self, key, idx): return self.min_state[key][idx]
    def get_max_state(self, key, idx): return self.max_state[key][idx]
    def get_min_input(self, key, idx): return self.min_input[key][idx]
    def get_max_input(self, key, idx): return self.max_input[key][idx]
    # get min and max for additional data
    def get_min_data(self, key, idx): return self.min_data[key][idx]
    def get_max_data(self, key, idx): return self.max_data[key][idx]

# TODO: add a more detailed comment about the flow
# TODO: add spawn drone --> single integrator but with drone icon
# TODO: add spawn unicycle --> use mobile robot icon
# TODO: visualizer 3D --> derive from Environment

class Environment: # Base class focusing on data access and logging
    def __init__(self):
        self.n = 0        
        # For Tracking robots dynamics
        self.robot_dyn = [None] * self.n
        self.data_log = [None] * self.n

    def spawn_robots( self, positions, Ts = None, model_name='SingleIntegrator'):
        n = len(positions)
        self.n += n
        self.robot_dyn += [None] * n
        self.data_log += [None] * n

        if Ts is None: Ts = 0.02 # By default consider 20ms update rate
        # Initialize robot dynamics and datalogger
        for i in range(self.n):
            if model_name == 'SingleIntegrator':
                self.robot_dyn[i] = SingleIntegrator(Ts, positions[i])
            elif model_name == 'Unicycle':
                self.robot_dyn[i] = Unicycle(Ts, positions[i])
            elif model_name == 'SI_DroneVision':
                self.robot_dyn[i] = SI_DroneVision(Ts, positions[i])
            else: exit() # not supported

            self.data_log[i] = DataLogger()
            # Initialize position data
            state = self.robot_dyn[i].get_state()
            input = self.robot_dyn[i].get_input()
            dot_state = self.robot_dyn[i].get_dot_state()
            self.data_log[i].update_history(state, dot_state, input, Ts)        

    def spawn_single_integrator(self, positions, Ts = None):
        # assuming positions is n array of 2 (or 3) dimensional array
        n = len(positions)
        self.n += n
        self.robot_dyn += [None] * n
        self.data_log += [None] * n

        if Ts is None: Ts = 0.02 # By default consider 20ms update rate
        # Initialize robot dynamics and datalogger
        for i in range(self.n):
            self.robot_dyn[i] = SingleIntegrator(Ts, positions[i])
            self.data_log[i] = DataLogger()
    
            # Initialize position data
            state = self.robot_dyn[i].get_state()
            input = self.robot_dyn[i].get_input()
            dot_state = self.robot_dyn[i].get_dot_state()
            self.data_log[i].update_history(state, dot_state, input, Ts)

    def spawn_unicycle(self, positions, Ts = None):
        # assuming positions is n array of 2 (or 3) dimensional array
        n = len(positions)
        self.n += n
        self.robot_dyn += [None] * n
        self.data_log += [None] * n

        if Ts is None: Ts = 0.02 # By default consider 20ms update rate
        # Initialize robot dynamics and datalogger
        for i in range(self.n):
            self.robot_dyn[i] = Unicycle(Ts, positions[i])
            self.data_log[i] = DataLogger()
    
            # Initialize position data
            state = self.robot_dyn[i].get_state()
            input = self.robot_dyn[i].get_input()
            dot_state = self.robot_dyn[i].get_dot_state()
            self.data_log[i].update_history(state, dot_state, input, Ts)

    # WRAPPER to access ROBOT DATA
    def getRobotState(self, ID): return self.robot_dyn[ID].get_state()
    def getRobotNum(self): return self.n

    def sendRobotCommand(self, ID, u): self.robot_dyn[ID].set_input(u, "u")

    def set_lookUpAhead_param(self, ID, ell): self.robot_dyn[ID].set_lookUpAhead_param(ell)
    def send_unicycle_command(self, ID, vlin, omega): self.robot_dyn[ID].set_input_VOmg(input_V=vlin, input_omg=omega)

    def update_simulation(self):
        for i in range(self.n):
            # IMPORTANT: update dynamics and save to trajectory
            state = self.robot_dyn[i].step_dynamics() 
            input = self.robot_dyn[i].get_input()
            dot_state = self.robot_dyn[i].get_dot_state()
            dt = self.robot_dyn[i].get_Ts()
            self.data_log[i].update_history(state, dot_state, input, dt)


class Visualizer2D(Environment): # Focusing Visualizer on 2D plot
    def __init__(self):
        super().__init__() 

        # For plotting
        self.field = [None] * 3 # with axis 0:x, 1:y, and 2:z
        self.additionalPlot = []
        self.addSingleLargePlot = False # as big as original 2D plot
        self.colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Default viewer
        # By default 2D plot is always on. TODO: support removal if needed
        self.showPosXY = True
        self.showVelXY = True
        self.showUserAdded = True

        # Store the plot option (use icon or not)
        self.draw_with_mobile_robot_icon = False

    def setPlot_withIcon(self, mode='omnidirectional'):
        self.draw_with_mobile_robot_icon = True
        if mode == 'omnidirectional': self.icon_id = 3
        elif mode == 'unicycle': self.icon_id = 2
        else: self.draw_with_mobile_robot_icon = False
        self.moro_patch = {}
        #  TODO: revise the visualization for more icon to use

    # PLOTTING, with axis 0:x, 1:y, and 2:z
    def setPlotField(self, axis, field): self.field[axis] = field
    def setPlotOnly2D(self):
        self.showPosXY = False
        self.showVelXY = False
        self.showUserAdded = False

    def registerAdditionalPlot(self, yaxisname):
        # Now only work for plotting value over time
        self.additionalPlot.append( {'label':yaxisname} )

    # Initialize the first plot
    def plot2DTraj(self):        
        # By default 2D plot always on and take 2 space
        colNum, rowNum = 2, 2 
        addPlotNum = 0
        if self.showPosXY: colNum += 1
        if self.showVelXY: colNum += 1
        if self.addSingleLargePlot: colNum += 2
        if self.showUserAdded:
            # Get information on registered Additional Plot
            addPlotNum = len(self.additionalPlot)
            rowNum += (addPlotNum+colNum-1) // colNum

        # TODO: automatically allocate the plotting grid. 

        # GridSize Allocation
        fig = plt.figure(figsize=(3*colNum-1, 3*rowNum-1), dpi= 100)
        self.gs = GridSpec( rowNum, colNum, figure=fig)

        # Plot 2D visualization
        self.ax_2D = fig.add_subplot(self.gs[0:2,0:2]) # Always on
        self.ax_2D.set(xlabel="x [m]", ylabel="y [m]")
        self.ax_2D.set_aspect('equal', adjustable='box', anchor='C')
        # For plotting
        self.pl2DTraj = [None] * self.n
        self.pl2DPos = [None] * self.n
        self.pl2DTime = [None]

        colSel, rowSel = 2, 2
        if self.addSingleLargePlot:
            # Just prepare the place and the axes, let the main program fill what is needed
            self.ax_LargePlot = fig.add_subplot(self.gs[0:2,colSel:colSel+2])
            colSel += 2

        if self.showPosXY:
            self.ax_x_traj = fig.add_subplot(self.gs[0,colSel])
            self.ax_y_traj = fig.add_subplot(self.gs[1,colSel])
            self.ax_x_traj.set(xlabel="t [s]", ylabel="x [m]")
            self.ax_y_traj.set(xlabel="t [s]", ylabel="y [m]")
            self.ax_x_traj.grid()
            self.ax_y_traj.grid()
            # For plotting
            self.pl_x = [None] * self.n
            self.pl_y = [None] * self.n
            colSel += 1

        if self.showVelXY:
            self.ax_u_x = fig.add_subplot(self.gs[0,colSel])
            self.ax_u_y = fig.add_subplot(self.gs[1,colSel])
            self.ax_u_x.set(xlabel="t [s]", ylabel="u_x [m/s]")
            self.ax_u_y.set(xlabel="t [s]", ylabel="u_y [m/s]")
            self.ax_u_x.grid()
            self.ax_u_y.grid()
            # For plotting
            self.pl_ux = [None] * self.n
            self.pl_uy = [None] * self.n
            colSel += 1

        # Currently add below 2D plot according to the number of column
        # TODO: add on the right side up to 2 col if any showPosXY or showVelXY is False
        if self.showUserAdded:
            self.ax_add = {} 
            self.pl_add = [None]*addPlotNum
            for i in range(addPlotNum):
                self.ax_add[i] = fig.add_subplot(self.gs[ (rowSel+(i//colNum)), (i%colNum)])
                self.ax_add[i].set(xlabel="t [s]", ylabel=self.additionalPlot[i]['label'])
                # For plotting
                self.ax_add[i].grid()
                self.pl_add[i] = [None] * self.n

        self.fig = fig
        plt.tight_layout()

        # Get Data and plot the first frame
        for i in range(self.n):   
            # Visualize Trajectory
            t = self.data_log[i].time
            qx = self.data_log[i].state["q"][:,0] # Always required for 2D plot
            qy = self.data_log[i].state["q"][:,1] # Always required for 2D plot
            self.pl2DTraj[i], = self.ax_2D.plot(qx, qy, '--', color=self.colorList[i])

            if self.draw_with_mobile_robot_icon: # use mobile robot icon
                self.moro_patch[i] = None
                self.draw_icon( i, np.array([qx[-1], qy[-1], 0]), arrow_col=self.colorList[i])
            else: # use simple x marker
                self.pl2DPos[i], = self.ax_2D.plot(qx[-1], qy[-1], color=self.colorList[i], marker='X', markersize=10)

            if self.showPosXY:
                self.pl_x[i], = self.ax_x_traj.plot(t, qx, color=self.colorList[i])
                self.pl_y[i], = self.ax_y_traj.plot(t, qy, color=self.colorList[i])

            # Visualize the changes of Input
            if self.showVelXY:
                ux = self.data_log[i].input["u"][:,0]
                uy = self.data_log[i].input["u"][:,1]
                self.pl_ux[i], = self.ax_u_x.plot(t, ux, color=self.colorList[i])
                self.pl_uy[i], = self.ax_u_y.plot(t, uy, color=self.colorList[i])
        # Can use the first t data
        self.pl2DTime = self.ax_2D.text(0.78, 0.99, 't = 0 s', color = 'k', fontsize='large', 
            horizontalalignment='left', verticalalignment='top', transform = self.ax_2D.transAxes)
        #self.pl2DTime = self.ax_2D.text(0.78, 0.01, 't = 0 s', color = 'k', fontsize='large', 
        #    horizontalalignment='left', verticalalignment='bottom', transform = self.ax_2D.transAxes)

        if self.showUserAdded:
            # Loop over each plot
            for i in range(addPlotNum):
                y_label = self.additionalPlot[i]['label']
                # Loop over each agent
                for j in range(self.n):
                    t = self.data_log[j].time
                    yData = self.data_log[j].data[y_label]

                    iter_k = yData.shape[1]
                    self.pl_add[i][j] = [None] * iter_k
                    for k in range(iter_k): # Loop over each column in the data
                        self.pl_add[i][j][k], = self.ax_add[i].plot(t, yData[:,k], color=self.colorList[j])

    # Update plot each iterations
    def update_plot2DTraj(self, shortTail=None):
        m = 0
        # assuming the assigned value is always positive int
        if shortTail is not None: m = min(0, -1-shortTail)

        for i in range(self.n):
            t = self.data_log[i].time
            qx = self.data_log[i].state["q"][:,0]
            qy = self.data_log[i].state["q"][:,1]
            ux = self.data_log[i].input["u"][:,0]
            uy = self.data_log[i].input["u"][:,1]
            assert np.size(qx) == np.size(t), f"Mismatched size. qx: {qx} and t: {t}"
            assert np.size(qy) == np.size(t), f"Mismatched size. qy: {qy} and t: {t}"

            self.pl2DTraj[i].set_data(qx[m:], qy[m:])
            if self.draw_with_mobile_robot_icon: # use wheeled robot icon
                theta = 0.0
                state = np.zeros(3)
                if self.icon_id == 3: theta = np.arctan2(uy[-1], ux[-1]) #  TODO: revise the icon display to use proper pose data
                if self.icon_id == 2: theta = self.data_log[i].state["theta"][-1]
                state[0], state[1], state[2] = qx[-1], qy[-1], theta
                self.draw_icon( i, state, arrow_col=self.colorList[i])
            else: # update the x marker
                self.pl2DPos[i].set_data(qx[-1], qy[-1])

            if self.showPosXY:
                self.pl_x[i].set_data(t, qx)
                self.pl_y[i].set_data(t, qy)

            if self.showVelXY:
                assert np.size(ux) == np.size(t), f"Mismatched size. ux: {qx} and t: {t}"
                assert np.size(uy) == np.size(t), f"Mismatched size. uy: {qy} and t: {t}"
                
                self.pl_ux[i].set_data(t, ux)
                self.pl_uy[i].set_data(t, uy)

        t_window = 5
        cur_t = t[-1]
        self.pl2DTime.set_text('t = '+f"{cur_t:.1f}"+' s')
        if (cur_t < t_window): t_range = (-0.1, t_window+0.1)
        else: t_range = (cur_t-(t_window+0.1), cur_t+0.1)

        min_hist_x  = min( [self.data_log[i].get_min_state("q", 0) for i in range(self.n)] )
        min_hist_y  = min( [self.data_log[i].get_min_state("q", 1) for i in range(self.n)] )
        max_hist_x  = max( [self.data_log[i].get_max_state("q", 0) for i in range(self.n)] )
        max_hist_y  = max( [self.data_log[i].get_max_state("q", 1) for i in range(self.n)] )
        if self.field[0] is not None: self.ax_2D.set(xlim=(self.field[0][0]-0.1, self.field[0][1]+0.1))
        if self.field[1] is not None: self.ax_2D.set(ylim=(self.field[1][0]-0.1, self.field[1][1]+0.1))

        if self.showPosXY:
            self.ax_x_traj.set(xlim=t_range, ylim=( min_hist_x-0.1, max_hist_x+0.1))
            self.ax_y_traj.set(xlim=t_range, ylim=( min_hist_y-0.1, max_hist_y+0.1))

        if self.showVelXY:
            min_input_ux = min( [self.data_log[i].get_min_input("u", 0) for i in range(self.n)] )
            min_input_uy = min( [self.data_log[i].get_min_input("u", 1) for i in range(self.n)] )
            max_input_ux = max( [self.data_log[i].get_max_input("u", 0) for i in range(self.n)] )
            max_input_uy = max( [self.data_log[i].get_max_input("u", 1) for i in range(self.n)] )
            self.ax_u_x.set(xlim=t_range, ylim=( min_input_ux-0.1, max_input_ux+0.1))
            self.ax_u_y.set(xlim=t_range, ylim=( min_input_uy-0.1, max_input_uy+0.1))

        if self.showUserAdded:
            # Loop over each plot
            for i in range(len(self.additionalPlot)):
                y_label = self.additionalPlot[i]['label']
                # Loop over each agent
                for j in range(self.n):
                    t = self.data_log[j].time
                    yData = self.data_log[j].data[y_label]

                    iter_k = yData.shape[1]
                    for k in range(iter_k): # Loop over each column in the data
                        self.pl_add[i][j][k].set_data(t, yData[:,k])
    
                min_hist_h = np.nanmin( [self.data_log[l].get_min_data(y_label, 0) for l in range(self.n)] )
                max_hist_h = np.nanmax( [self.data_log[l].get_max_data(y_label, 0) for l in range(self.n)] )
                self.ax_add[i].set(xlim=t_range, ylim=( min_hist_h-0.1, max_hist_h+0.1))

    # Additional plotting function
    def drawCircle(self, xy, rad, col = 'k'):
        objCircle = plt.Circle( xy, rad, color = col)
        self.ax_2D.add_patch(objCircle)

    def drawMovingGoalPoints(self, xy, rad=0.03, col = 'k'):
        self.pl_mg, = self.ax_2D.plot(xy[:,0], xy[:,1], 'r.')
    def updateMovingGoalPoints(self, xy):
        self.pl_mg.set_data(xy[:,0], xy[:,1])

    def drawMovingEllipseFormation(self, name, pos, major_l, minor_l, theta=0., alpha=0.2, col = 'k'):
        if not hasattr(self, 'pl_me'):
            self.pl_me = {}
        self.pl_me[name] = Ellipse((pos[0], pos[1]), major_l, minor_l, angle=theta, alpha=alpha)
        self.ax_2D.add_artist(self.pl_me[name])

    def updateMovingEllipseFormation(self, name, xy, theta):
        self.pl_me[name].center = (xy[0], xy[1])
        self.pl_me[name].angle = np.rad2deg(theta)

    # OPTIONAL PLOT, not necessary but provide nice view in simulation
    #-----------------------------------------------------------------
    def draw_icon(self, id, robot_state, arrow_col = 'b'): # draw mobile robot as an icon
        # Extract data for plotting
        px = robot_state[0]
        py = robot_state[1]
        th = robot_state[2]
        # Basic size parameter
        scale = 1
        body_rad = 0.08 * scale # m
        wheel_size = [0.1*scale, 0.02*scale] 
        arrow_size = body_rad
        # left and right wheels anchor position (bottom-left of rectangle)
        if self.icon_id == 2: thWh = [th+0., th+np.pi] # unicycle
        else: thWh = [ (th + i*(2*np.pi/3) - np.pi/2) for i in range(3)] # default to omnidirectional
        thWh_deg = [np.rad2deg(i) for i in thWh]
        wh_x = [ px - body_rad*np.sin(i) - (wheel_size[0]/2)*np.cos(i) + (wheel_size[1]/2)*np.sin(i) for i in thWh ]
        wh_y = [ py + body_rad*np.cos(i) - (wheel_size[0]/2)*np.sin(i) - (wheel_size[1]/2)*np.cos(i) for i in thWh ]
        # Arrow orientation anchor position
        ar_st= [px, py] #[ px - (arrow_size/2)*np.cos(th), py - (arrow_size/2)*np.sin(th) ]
        ar_d = (arrow_size*np.cos(th), arrow_size*np.sin(th))
        # initialized unicycle icon at the center with theta = 0
        if self.moro_patch[id] is None: # first time drawing
            self.moro_patch[id] = [None]*(2+len(thWh))
            self.moro_patch[id][0] = self.ax_2D.add_patch( plt.Circle( (px, py), body_rad, color='#AAAAAAAA') )
            self.moro_patch[id][1] = self.ax_2D.quiver( ar_st[0], ar_st[1], ar_d[0], ar_d[1], 
                scale_units='xy', scale=1, color=arrow_col, width=0.1*arrow_size)
            for i in range( len(thWh) ):
                self.moro_patch[id][2+i] = self.ax_2D.add_patch( plt.Rectangle( (wh_x[i], wh_y[i]), 
                    wheel_size[0], wheel_size[1], angle=thWh_deg[i], color='k') )
        else: # update existing patch
            self.moro_patch[id][0].set( center=(px, py) )
            self.moro_patch[id][1].set_offsets( ar_st )
            self.moro_patch[id][1].set_UVC( ar_d[0], ar_d[1] )
            for i in range( len(thWh) ):
                self.moro_patch[id][2+i].set( xy=(wh_x[i], wh_y[i]) )
                self.moro_patch[id][2+i].angle = thWh_deg[i]