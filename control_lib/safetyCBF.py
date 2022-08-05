import numpy as np
import cvxopt


"""  Safety with Static Obstacle """ 
# h = norm2( pos - obs )^2 - norm2(ds)^2 > 0
def calcConstraint_StaticObst(pos, obs, ds, gamma=10, power=1):
    vect = pos - obs
    h_func = np.power(np.linalg.norm(vect), 2) - np.power(ds, 2)
    # -(dh/dpos)^T u < gamma(h)
    return -2*vect, gamma*np.power(h_func, power), h_func

"""  Safety with Dynamic Obstacle """ 
# h = norm2( pos - obs )^2 - norm2(ds)^2 > 0
def calcConstraint_DynamicObst(pos, obs, ds, obs_vel=np.zeros(3), gamma=10, power=1):
    vect = pos - obs
    h_func = np.power(np.linalg.norm(vect), 2) - np.power(ds, 2)
    # Compute the contraint formulation
    temp_G = -2*vect
    temp_h = gamma * np.power(h_func, power)
    # implementing TV-CBF
    # -(dh/dpos)^T u < gamma(h) + (dh/dobs)^T obs_vel
    corr = np.dot( -2*vect, obs_vel )
    return temp_G, temp_h + corr, h_func

"""  Safe Area with Boundary """ 
# h = ( axis_radius )^2 - ( pos_axis - center_axis )^2 > 0
# TODO: update for arbitrary line (if needed)
def calcConstraint_Bounds(axis_pos, center, ds, gamma=10, power=1): 
    vect = axis_pos - center
    h_func = np.power(ds, 2) - np.power(vect, 2)
    # -(dh/dpos)^T u < gamma(h)
    return 2*vect, gamma*np.power(h_func, power), h_func


"""  Safety in Maintaining Distance (within a certain tolerance epsilon) """ 
# h = norm2( ds + epsilon )^2 - norm2( pos - obs )^2 > 0
def calcConstraint_upperDistance(pos, obs, ds_pe, gamma=10, power=1):
    vect = pos - obs
    h_func = np.power(ds_pe, 2) - np.power(np.linalg.norm(vect), 2)
    # -(dh/dpos)^T u < gamma(h)
    return 2*vect, gamma*np.power(h_func, power), h_func
# h = norm2( pos - obs )^2 - norm2( ds - epsilon )^2 > 0
def calcConstraint_lowerDistance(pos, obs, ds_me, gamma=10, power=1):
    vect = pos - obs
    h_func = np.power(np.linalg.norm(vect), 2) - np.power(ds_me, 2)
    # -(dh/dpos)^T u < gamma(h)
    return -2*vect, gamma*np.power(h_func, power), h_func


"""  Safety in Avoiding Moving Elipse """ 
# h = norm2( ellipse*[pos - obs] )^2 - 1 > 0
def calcConstraint_EllipseObst(pos, obs, theta, major_l, minor_l, gamma=10, power=1):
    # TODO: assert a should be larger than b (length of major axis vs minor axis)
    vect = pos - obs # compute vector towards pos from centroid
    # rotate vector by -theta (counter the ellipse angle)
    # then skew the field due to ellipse major and minor axis
    # the resulting vector should be grater than 1
    # i.e. T(skew)*R(-theta)*vec --> then compute L2norm square
    ellipse = np.array([[2./major_l, 0, 0], [0, 2./minor_l, 0], [0, 0, 1]]) \
        @ np.array([[np.cos(-theta), -np.sin(-theta), 0], [np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    h_func = np.power(np.linalg.norm( ellipse @ vect.T ), 2) - 1
    # -(dh/dpos)^T u < gamma(h)
    # -(2 vect^T ellipse^T ellipse) u < gamma(h)
    G = -2*vect @ ( ellipse.T @ ellipse )
    return G, gamma*np.power(h_func, power), h_func


class safetyCBF():
    def __init__(self, sensingRange = None):
        self.currentPos = np.zeros(3) # will be updated in each calculation
        self.sensingRange = sensingRange
        self.staticObs = [] # data will be saved as dictionary
        self.dynamicObs = {} 
        self.useTVCBF = True
        self.safeBounds = [None]*3 # Box safe boundary
        self.formationLink = {} 
        self.ellipseObs = {}
        self.velocity_bound = None
        self.unicycle_vel_bound = None
        self.cur_theta = None


    """  Data: Safety with Static Obstacle """ 
    def isStaticObsPresent(self): return len(self.staticObs) > 0
    def registerStaticObs(self, position, radius = 0, gamma=10, power=1):
        self.staticObs.append({"rad": radius, "pos": position, "gam":gamma, "pow":power})


    """  Data: Safety with Dynamic Obstacle """ 
    def isDynamicObsPresent(self): return self.dynamicObsRadius > 0
    def registerDynamicObsRadius( self, radius):  
        self.dynamicObsRadius = radius # Note: setting zero is turning the function off

    def registerDynamicObs( self, name, radius, gamma=10, power=1):
        # assign 5 data point to save
        posAr = np.empty([5,3])
        posAr[:] = np.nan
        self.dynamicObs[name] = {"rad":radius, "posArray":posAr, "pos":posAr[0], "vel":np.nan, "gam":gamma, "pow":power}
        # initialize both pos and vel value as nan (unassigned / unavailable)
        # Note: setting radius to zero is turning the function off

    def updateDynamicObs( self, name, pos, dt):
        posAr = self.dynamicObs[name]["posArray"]
        posArRoll = np.roll(posAr, 1, axis=0) 
        posArRoll[0,:] = np.nan
        temp_vel = np.nan

        # assuming pos is awlays a valid value
        if self.useTVCBF:
            if self.isObj_withinSensingRange(pos): # within sensing range
                # update information
                posArRoll[0] = pos

                if np.isnan(posAr).any() or np.isnan(posArRoll).any(): # object just entered sensing region
                    temp_vel = np.zeros(3) # don't over-estimate
                else: # valid old_pos, estimate velocity
                    # temp_vel = (pos - old_pos)/dt 
                    # TODO: consider a more sophisticated form of filter
                    # https://dsp.stackexchange.com/questions/9498/given-position-measurements-how-to-estimate-velocity-and-acceleration
                    # https://dsp.stackexchange.com/questions/8860/kalman-filter-for-position-and-velocity-introducing-speed-estimates/8869#8869 
                    # Weighted Averaging of Velocity from 5 position
                    w = np.array([[0.7, 0.2, 0.1, 0.0, 0.0]]) # TODO: This value is really sensitive
                    #w = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) # Turn OFF TVCBF
                    #w = np.array([[1., 0., 0., 0., 0.]])
                    velAr = (posArRoll - posAr) / dt
                    mul = w @ velAr
                    temp_vel = mul[0]
        else:
            posArRoll[0] = pos
            temp_vel = np.zeros(3) # this will let the correction term into 0

        self.dynamicObs[name]["posArray"] = posArRoll
        self.dynamicObs[name]["pos"] = posArRoll[0]
        self.dynamicObs[name]["vel"] = temp_vel


    """  Data: Safe Area within Boundary """ 
    def registerBound(self, axis, bounds): 
        b = [min(bounds), max(bounds)]
        c = 0.5*sum(b)
        self.safeBounds[axis] = {"c": c, "r": (b[1] - c), "b":b}


    """  Data: Safety in Maintaining Distance (within a certain tolerance epsilon) """ 
    def registerFormationLink( self, name, radius, epsilon, gamma=10, power=1):
        self.formationLink[name] = {"rad":radius, "pos":np.nan, "epsilon":epsilon, "gam":gamma, "pow":power}
        # initialize both pos value as nan (unassigned / unavailable)

    def updateFormationLink( self, name, pos): self.formationLink[name]["pos"] = pos


    """  Data: Safety in Avoiding Ellipse obstacle """ 
    def registerEllipseObs( self, name, major_l, minor_l, gamma=10, power=1):
        self.ellipseObs[name] = {"pos":np.nan, "theta":np.nan, "major_l":major_l, "minor_l":minor_l, "gam":gamma, "pow":power}
        # initialize both pos value as nan (unassigned / unavailable)

    def updateEllipseObs( self, name, pos, theta): 
        self.ellipseObs[name]["pos"] = pos
        self.ellipseObs[name]["theta"] = theta


    """  Data: Linear Velocity bound """ 
    def register_velocity_bound( self, speed_limit):
        self.velocity_bound = speed_limit
    def compute_velocity_constraints( self ):
        G = np.vstack((np.eye(3), -np.eye(3)))
        h = np.ones([6, 1]) * self.velocity_bound
        return G, h

    """  Data: Unicycle Velocity bound """ 
    def register_unicycle_vel_bound( self, vlin_max, vang_max, ell):
        if vlin_max > 0 and vang_max > 0 and ell > 0 :
            self.unicycle_vel_bound = {'v_lin':vlin_max, 'v_ang':vang_max, 'ell': ell}
    def compute_unicycle_vel_constraints( self ):
        Ml = np.array([[1, 0, 0], [0, 1/self.unicycle_vel_bound['ell'], 0], [0, 0, 1]])
        Mth = np.array([[np.cos(self.cur_theta), np.sin(self.cur_theta), 0], 
            [-np.sin(self.cur_theta), np.cos(self.cur_theta), 0], [0, 0, 1]])
        mult = Ml @ Mth
        G = np.vstack((mult[:2], -mult[:2]))
        h = np.array([[ self.unicycle_vel_bound['v_lin'], self.unicycle_vel_bound['v_ang'], self.unicycle_vel_bound['v_lin'], self.unicycle_vel_bound['v_ang'] ]]).transpose()
        return G, h
        
    def update_current_theta( self, theta ):
        self.cur_theta = theta


    """ Function for sensing Range """
    def isObj_withinSensingRange(self, obj): 
        if self.sensingRange is not None:
            return np.linalg.norm(self.currentPos - obj) < self.sensingRange
        else: return True # No sensing range (compute all)

    def isBound_withinSensingRange(self, idx): # idx >> 0: x-axis, 1: y-axis, 2: z-axis
        if self.safeBounds[idx] is not None:
            if self.sensingRange is not None:
                cp = self.currentPos[idx] # get specific axis data from position
                bd = self.safeBounds[idx]["b"] # get bounds data
                return min( abs(cp - bd[0]), abs(cp - bd[1]) ) < self.sensingRange
            else: return True # Exist Bounds but No sensing range (compute)
        else: return False # No bounds


    """ Compute CBF Safety Optimization """
    def computeSafeController(self, current_q, u_nom):
        # Update position data
        self.currentPos = current_q

        # IMPLEMENTATION OF Control Barrier Function
        # Minimization
        P_mat = 2 * cvxopt.matrix( np.eye(3), tc='d')
        q_mat = -2 * cvxopt.matrix( u_nom, tc='d')

        row = 0
        # Compute total constraints & store index
        staticIdx = []
        h_staticObs = [np.nan]*len(self.staticObs)
        for i in range( len(self.staticObs) ):
            if self.isObj_withinSensingRange(self.staticObs[i]["pos"]):
                staticIdx.append(i)
        row += len( staticIdx )

        dynamicIdx = []
        h_dynamicObs = [np.nan]*len(self.dynamicObs.keys())
        for i, val in self.dynamicObs.items():
            if (val["rad"] > 0) and (not np.isnan( val["pos"] ).any()): # valid position within range
                if self.isObj_withinSensingRange( val["pos"] ): # within range
                    dynamicIdx.append(i)
        row += len( dynamicIdx )

        boundExists = []
        h_bound = [np.nan]*3
        for i in range(3):
            if self.isBound_withinSensingRange(i): 
                boundExists.append(i)
        row += len(boundExists)

        ellipseIdx = []
        h_ellipseObs = [np.nan]*len(self.ellipseObs)
        for i, val in self.ellipseObs.items():
            if (not np.isnan( val["theta"] ).any()) and (not np.isnan( val["pos"] ).any()): # valid position within range
                if self.isObj_withinSensingRange( val["pos"] ): # within range
                    ellipseIdx.append(i)
        row += len( ellipseIdx )

        formLinkIdx = []
        h_formation = [np.nan]*( 2*len(self.formationLink.keys()) )
        for i, val in self.formationLink.items():
            if (val["rad"] > 0) and (not np.isnan( val["pos"] ).any()): # valid position within range
                if self.isObj_withinSensingRange( val["pos"] ): # within range
                    formLinkIdx.append(i)
        row += 2*len( formLinkIdx )

        if (self.velocity_bound is not None) and (self.velocity_bound > 0): row += 6
        if (self.unicycle_vel_bound is not None): row += 4

        # initialize G and h, Then fill it afterwards
        G = np.zeros([row, 3])
        h = np.zeros([row, 1])

        shift = 0
        # Inequality constraint (CBF) -- Static Obstacle
        for i, idx in enumerate(staticIdx):
            ds = self.staticObs[idx]["rad"] # Radius avoidance
            obst_pos = self.staticObs[idx]["pos"] # Obstacle Position
            gam = self.staticObs[idx]["gam"]
            pow = self.staticObs[idx]["pow"]
            # calculate constraints
            G[shift + i], h[shift + i], h_staticObs[idx] = \
                calcConstraint_StaticObst(current_q, obst_pos, ds, gam, pow)
        shift += len( staticIdx )

        # Inequality constraint (CBF) -- Dyamic Obstacle
        for i, idx in enumerate(dynamicIdx):
            ds = self.dynamicObs[idx]["rad"] # Radius avoidance
            obst_pos = self.dynamicObs[idx]["pos"] # Obstacle Position
            obst_vel = self.dynamicObs[idx]["vel"]
            gam = self.dynamicObs[idx]["gam"]
            pow = self.dynamicObs[idx]["pow"]
            # calculate constraints
            G[shift + i], h[shift + i], h_dynamicObs[i] = \
                calcConstraint_DynamicObst(current_q, obst_pos, ds, obst_vel, gam, pow)
        shift += len( dynamicIdx )

        # Inequality constraint (CBF) -- Safe Area
        for i, idx in enumerate(boundExists):
            cent = self.safeBounds[idx]["c"]
            ds = self.safeBounds[idx]["r"]
            # idx >> 0: x-axis, 1: y-axis, 2: z-axis
            G[shift + i][idx], h[shift + i], h_bound[idx] = \
                calcConstraint_Bounds(current_q[idx], cent, ds)
        shift += len( boundExists )

        # Inequality constraint (CBF) -- Ellipse Obstacle
        for i, idx in enumerate(ellipseIdx):
            major_l = self.ellipseObs[idx]["major_l"]
            minor_l = self.ellipseObs[idx]["minor_l"]
            obst_pos = self.ellipseObs[idx]["pos"] # Obstacle Position
            obst_theta = self.ellipseObs[idx]["theta"] # Ellipse angle
            gam = self.ellipseObs[idx]["gam"]
            pow = self.ellipseObs[idx]["pow"]
            # calculate constraints
            G[shift + i], h[shift + i], h_ellipseObs[i] = \
                calcConstraint_EllipseObst(current_q, obst_pos, obst_theta, major_l, minor_l, gam, pow)
        shift += len( ellipseIdx )

        # Inequality constraint (CBF) -- Maintaining Formation
        for i, idx in enumerate(formLinkIdx):
            ds = self.formationLink[idx]["rad"] # Radius avoidance
            q_other = self.formationLink[idx]["pos"] # Obstacle Position
            epsilon = self.formationLink[idx]["epsilon"]
            gam = self.formationLink[idx]["gam"]
            pow = self.formationLink[idx]["pow"]
            # calculate constraints
            G[shift + (2*i)], h[shift + (2*i)], h_formation[(2*i)] = \
                calcConstraint_upperDistance(current_q, q_other, (ds + epsilon), gam, pow)
            G[shift + (2*i)+1], h[shift + (2*i)+1], h_formation[(2*i)+1] = \
                calcConstraint_lowerDistance(current_q, q_other, (ds - epsilon), gam, pow)
        shift += 2*len( formLinkIdx )

        if (self.velocity_bound is not None) and (self.velocity_bound > 0): 
            G[shift:shift+6], h[shift:shift+6] = self.compute_velocity_constraints()
            shift += 6

        if (self.unicycle_vel_bound is not None):
            G[shift:shift+4], h[shift:shift+4] = self.compute_unicycle_vel_constraints()
            shift += 4


        # Resize the G and H into appropriate matrix for optimization
        G_mat = cvxopt.matrix( G, tc='d') 
        h_mat = cvxopt.matrix( h, tc='d')

        # Solving Optimization
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P_mat, q_mat, G_mat, h_mat, verbose=False)

        if sol['status'] == 'optimal':
            # Get solution + converting from cvxopt base matrix to numpy array
            u_star = np.array([sol['x'][0], sol['x'][1], sol['x'][2]])
        else: 
            print( 'WARNING QP SOLVER id-' + str(i) + ' status: ' + sol['status'] + ' --> use nominal instead' )
            u_star = u_nom.copy()


        ret_h = {}
        ret_h['h_staticObs'] = h_staticObs
        ret_h['h_dynamicObs'] = h_dynamicObs
        ret_h['h_bound'] = h_bound
        ret_h['h_formation'] = h_formation
        ret_h['h_ellipseObs'] = h_ellipseObs

        # TODO: output h value and label
        return u_star, ret_h 