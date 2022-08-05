import numpy as np


# LOCAL FUNCTION
# ------------------------------------------------------------
def ensure_input1D(input, msg):
    # assert that the input is numpy array
    assert isinstance(input, np.ndarray), f"Assigned input {msg} should be numpy array. Actual input: {input}"
    # Flatten the input if it is not 1D array
    return input if input.ndim == 1 else input.flatten()

def ensure_xyz(input, label=None):
    msg = ('for ' + label) if isinstance(label, str) else ''
    temp = ensure_input1D(input, msg) # Assert that input is numpy array and Flatten if it is not 1D array
    if temp.size == 2: temp = np.append(temp, 0) # pad z = 0 if it is only x and y (i.e., size == 2)
    assert temp.size==3, f"Assigned input {msg} is preferably 1D array with 2 or 3 values. Actual input: {input}"
    # Assert that the size is 3 at the end
    return temp

def ensure_1D_ndim(input, ndim, label=None):
    msg = ('for ' + label) if isinstance(label, str) else ''
    temp = ensure_input1D(input, msg) # Assert that input is numpy array and Flatten if it is not 1D array
    assert temp.size==ndim, f"Assigned input {msg} is preferably 1D array with {ndim} values. Actual input: {input}"
    # Assert that the size is 3 at the end
    return temp

def ensure_scalar(input, msg):
    # assert that the input is numpy array
    assert np.size(input) == 1, f"Assigned scalar {msg} should only have single value. Actual input: {input}"
    # extract the input if not scalar
    return input if np.ndim(input) == 0 else input.item()

def ensure_theta(input, label=None):
    # Utilize ensure_xyz for initial checking >> ensuring numpy with 3 elements
    msg = ('for ' + label) if isinstance(label, str) else ''
    temp = ensure_scalar(input, msg) # ensure that it is a single scalar value
    assert (temp[2] >= 0) and (temp[2] <= 2*np.pi), f"Assigned theta {msg} is shoule be within 0 to 2pi. Actual input: {input}"
    return temp

#def ensure_VOmg(input, label=None):
#    msg = ('for ' + label) if isinstance(label, str) else ''
#    temp = ensure_input1D(input, msg) # Assert that input is numpy array and Flatten if it is not 1D array
#    assert temp.size==2, f"Assigned input {msg} is preferably 1D array with 2 values. Actual input: {input}"
#    return temp[0], temp[1]

# DYNAMICS BASECLASS >> This cannot be used stand alone
# -----------------------------------------------------------------------
class Dynamics:
    def __init__(self, dt):
        # Initialize the state and input
        self.dt = dt 
        self.input = {}
        self.state = {}
        self.dot_state = {}

    # NOTE: these two function belows need to be implemented in child class
    # def compute_dot_state(self): pass
    # def set_input(self, input): pass

    # Getting and setting the default time-sampling
    def get_Ts(self): return self.dt
    def set_Ts(self, Ts): self.dt = Ts

    # Function for getting the current input, state and dot_state
    def get_input(self, key=None): 
        return self.input if key is None else self.input[key]
    def get_state(self, key=None): 
        return self.state if key is None else self.state[key]
    def get_dot_state(self, key=None): 
        return self.dot_state if key is None else self.dot_state[key]

    # Update states for one time step, based on the nominal dynamics
    def step_dynamics(self, Ts = None):
        # Allow computation with varying time sampling, but only this instance
        # If not defined again in the next step it default back to existing self.dt
        # for permanent changes of dt use set_Ts function.
        dt = self.dt if Ts is None else Ts

        # Compute next state (based on nominal model)
        self.compute_dot_state()
        # Increment from past to present state
        for key, val in self.state.items(): 
            self.state[key] = val + dt*self.dot_state[key]

        return self.state


class SingleIntegrator(Dynamics):
    #
    # State: q = [q_x q_y q_z] 
    # Input: u = [u_x u_y u_z] 
    # Single Integrator Dynamics:
    #   dot(q) = u
    #
    # Note: 
    # in mathematical formulation, the q and u is usually presented as column vector (2x1)
    # But for this implementation we opted for row vector with 1D numpy array. 
    
    def __init__(self, dt, init_pos, init_vel=np.array([0., 0., 0.])):
        super().__init__(dt) 

        # Initialize the state and input
        # Ensure that the input is 1D array with 3 values (xyz)
        self.state["q"] = ensure_xyz(init_pos, '[SingleIntegrator init_pos]')
        self.set_input(init_vel)
        self.compute_dot_state()

    def compute_dot_state(self):
        # Initialize dot_state
        self.dot_state["q"] = self.input["u"]

    def set_input(self, input, key="u", check_input=True): 
        # NOTE this part is beneficial but checking this each time can take some time for the simulation.
        self.input[key] = ensure_xyz(input, '[SingleIntegrator input_vel]') if check_input else input


class SI_DroneVision(Dynamics):
    #
    # State: q = [q_x q_y q_z lambda_i] 
    # Input: u = [u_x u_y u_z u_lambda] 
    # Single Integrator Dynamics:
    #   dot(q) = u
    
    def __init__(self, dt, init_pos, init_vel=np.array([0., 0., 0., 0.])):
        super().__init__(dt) 

        # Initialize the state and input
        # Ensure that the input is 1D array with 4 values (xyzlambda)
        self.state["q"] = ensure_1D_ndim(init_pos, 4, '[SI_DroneVision init_pos]')
        self.set_input(init_vel)
        self.compute_dot_state()

    def compute_dot_state(self):
        # Initialize dot_state
        self.dot_state["q"] = self.input["u"]

    def set_input(self, input, key="u", check_input=True): 
        # NOTE this part is beneficial but checking this each time can take some time for the simulation.
        self.input[key] = ensure_1D_ndim(input, 4, '[SI_DroneVision input_vel]') if check_input else input


class Unicycle(Dynamics): # UNTESTED
    #
    # State: q = [q_x q_y theta] or [q_x q_y qz theta]
    # Input: u = [vlin omg] in paralel [u_x u_y u_z] 
    # Single Integrator Dynamics:
    #   dot(q_x) = vlin*cos(theta)
    #   dot(q_y) = vlin*sin(theta)
    #   dot(q_z) = 0
    #   dot(theta) = omg
    #
    # Note: 
    # in mathematical formulation, the q and u is usually presented as column vector (2x1)
    # But for this implementation we opted for row vector with 1D numpy array. 
    # There is no need for z values as we assum the robot lives in a flat world in the ground.
    
    def __init__(self, dt, 
            init_pos, init_theta=0., 
            init_vel=np.array([0., 0., 0.]), init_vlin=None, init_omg=None,
            ell=1.):
        
        super().__init__(dt) 

        # Ensure that the input is 1D array with 3 values (xyz)
        self.state["q"] = ensure_xyz(init_pos, '[Unicycle init_pos]')
        self.state["theta"] = ensure_scalar(init_theta, '[Unicycle init_theta]')
        self.lookAhead_l = ell

        if (init_vlin is not None) or (init_omg is not None): # if any of them is assigned
            # These value take precedence over init_vel. if one value is None, just set it to 0.
            input_v = init_vlin if init_vlin is not None else 0.
            input_o = init_omg if init_omg is not None else 0.
            self.set_input_VOmg(input_v, input_o)
        else:
            self.set_input(init_vel)
        self.compute_dot_state()


    def compute_dot_state(self):
        # Initialize dot_state
        self.dot_state["q"] = np.array([ 
            self.input["V"] * np.cos(self.state["theta"]),
            self.input["V"] * np.sin(self.state["theta"]),
            0.
        ])
        self.dot_state["theta"] = self.input["omg"]


    def set_lookUpAhead_param(self, ell): self.lookAhead_l = ell

    def set_input_VOmg(self, input_V, input_omg): # for now assume the input is numpy array with 2 values 
        # NOTE this part is beneficial but checking this each time can take some time for the simulation.
        self.input["V"] = ensure_scalar(input_V, '[Unicycle set_input_VOmg V]')
        self.input["omg"] = ensure_scalar(input_omg, '[Unicycle set_input_VOmg omg]')

        # Look up ahead Mapping (u_z remain 0.)
        #   u_x = V cos(theta) - l sin(theta)
        #   u_y = V sin(theta) + l cos(theta)
        theta = self.state["theta"]
        Mth = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Ml = np.array([[1, 0], [0, self.lookAhead_l]])
        self.input["u"][:2] = Mth @ Ml @ np.array([self.input["V"], self.input["omg"]])

        # self.input["u"][0] = self.input["V"]*np.cos(theta) - self.lookAhead_l*np.sin(theta)
        # self.input["u"][1] = self.input["V"]*np.sin(theta) + self.lookAhead_l*np.cos(theta)

    def set_input(self, input, key="u", check_input=True): 
        # NOTE this part is beneficial but checking this each time can take some time for the simulation.
        self.input[key] = ensure_xyz(input, '[SingleIntegrator input_vel]') if check_input else input

        # Inverse Look up ahead Mapping (u_z remain 0.)
        #   V = u_x cos(theta) + u_y sin(theta)
        #   omg = (- u_x sin(theta) + u_y cos(theta)) / l
        theta = self.state["theta"]

        # do SI to unicycle conversion
        Ml = np.array([[1, 0], [0, 1/self.lookAhead_l]])
        Mth = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        current_input = Ml @ Mth @ self.input["u"][:2]

        self.input["V"] = current_input[0]
        self.input["omg"] = current_input[1]

        # self.input["V"] = self.input["u"][0]*np.cos(theta) + self.input["u"][1]*np.sin(theta)
        # self.input["omg"] = (- self.input["u"][0]*np.sin(theta) + self.input["u"][1]*np.cos(theta))/self.lookAhead_l
