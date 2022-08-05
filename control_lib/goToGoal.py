import numpy as np

""" Implement simple P control. """
def Pcontrol(current_q, Pgain, ref):
    return Pgain*(ref - current_q)


def Pcontrol_TimeVarying(current_q, ref, v0=1., beta=1.):
    error = (ref - current_q)
    norm_er_pos = np.hypot(error[0], error[1])
    kP = v0*(1 - np.exp(-beta*norm_er_pos))/norm_er_pos
    return kP*error