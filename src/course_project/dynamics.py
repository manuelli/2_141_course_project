import scipy
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def make_default_params():
    params = dict()
    params['J'] = 0.0005
    params['T_1'] = 1/7.66
    params['T_2']  = 1/8.846
    params['b'] = 0.1
    params['T'] = params['T_2']
    params['transmission_loss'] = 0.05
    
    T = params['T']
    
    # rotor vibrates at this frequency when hit
    # equivalent to a spring
    omega_resonant_hz = 30.0
    omega_resonant_rad_s = 2 * np.pi * 30.0
    gearbox_stiffness_before_reduction = omega_resonant_rad_s**2 * params['J']

    k_s = gearbox_stiffness_before_reduction
    params['k_s'] = k_s
    
    
    # move to reflected parameters
    params['J'] = 1/T**2 * params['J']
    params['k_s'] = 1/T**2 * params['k_s']
    params['b'] = 1/T**2 * params['b']
        
    return params

def linear_system_tf_from_params(params):
    k_s = params['k_s']
    J = params['J']
    b = params['b']
    system = scipy.signal.lti(np.array([k_s]), [J, b, k_s])
    return system


def linear_system_ss_from_params(params):
    """
    Inputs are [tau, x_b]
    """
    k_s = params['k_s']
    J = params['J']
    b = params['b']
    
    A = np.array([[0, 1], [-k_s/J, -b/J]])
    B = np.array([[0,0], [1/J, -k_s/(J)]])
    C = np.array([k_s, 0])
    D = np.array([0, k_s])
    
    sys = scipy.signal.lti(A, B, C, D)
    return sys, [A, B, C, D]

def impedance_ss_system(params, controller_params):
    """
    controller_params is dict with keys ['k_p', 'k_d', 'x_0']
    Note we can only measure \theta, not x. So we implement the feedback using the 
    motor angles
    """
    _, [A,B,C,D] = linear_system_ss_from_params(params)
    
    
    k_p = controller_params['k_p']
    k_d = controller_params['k_d']
    J = params['J']
    # controller gain matrix
    # u = K [\theta, \dot{\theta}]
    K_vec = -1/J * np.array([k_p, k_d])
    K = np.zeros([2,2])
    K[1,:] = K_vec
    A_fb = A + K
    
    k_s = params['k_s']
    B = np.array([[0], [-k_s/J]])
    
    D = D[1]
    
    sys = scipy.signal.lti(A_fb, B, C, D)
    return sys, [A_fb,B,C,D]   



class SingleDofCoulombFriction(object):

    def __init__(self, params):
        self._params = params
        sys, [A,B,C,D] = linear_system_ss_from_params(params)
        self._A = A
        self._B = B
        self._C = C
        self._D = D

        self._J = params['J']
        self._k_s = params['k_s']
        self._b = params['b']
        self._alpha = params['transmission_loss']

        self.fixture_position_function = None
        self.motor_torque_function = None

        self._friction_tol = 1e-2

    def dydt(self, t, y):

        """
        Dynamics system
        y = [theta, \dot{theta}]
        """
        theta = y[0]
        theta_dot = y[1]
        x_b, x_b_dot = self.get_fixture_position(t)
        tau_m = self.get_motor_torque(t,y,x_b, x_b_dot)

        # a should be in the range [0,1]
        val = tau_m * theta_dot
        val = theta_dot
        a = 1/(2.0*self._friction_tol) *(np.clip(val, -self._friction_tol, self._friction_tol) + self._friction_tol)

        

        torque_multiplier = (1-a) * 1.0/(1-self._alpha) + a * (1-self._alpha)


        # print a
        # print "torque_multiplier", torque_multiplier

        # if (tau_m * theta_dot) >= 0:
        #     torque_multiplier = 1.0/(1-self._alpha)
        # else:
        #     torque_multiplier = (1-self._alpha)

        theta_ddot = 1/self._J * (tau_m - self._b * theta_dot - torque_multiplier * self._k_s * (theta + x_b))
        
        dydt = np.array([theta_dot, theta_ddot])

        return dydt

    def get_motor_torque(self, t, y, x_b, x_b_dot):
        return self.motor_torque_function(t, y, x_b, x_b_dot)

    def get_fixture_position(self, t):
        return self.fixture_position_function(t)

    def simulate(self, t_vec, y0=None, atol=None, rtol=None):
        if y0 is None:
            y0 = np.array([0,0])

        y = scipy.integrate.odeint(self.dydt, y0, t_vec, tfirst=True, atol=atol, rtol=rtol)

        return t_vec, y





    