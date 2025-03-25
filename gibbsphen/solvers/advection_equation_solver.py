import numpy as np

def advection_equation_solver(u_initial, c, h, k, num_time_steps):
    """
    Solves the 1D advection equation (u_t + c*u_x = 0) using RK4 method.
    
    Parameters:
    -----------
    u_initial : numpy.ndarray
        Initial values at spatial points
    c : float
        Wave speed
    h : float
        Spatial step size
    k : float
        Time step size
    num_time_steps : int
        Number of time steps to simulate
        
    Returns:
    --------
    u_solution : numpy.ndarray
        Solution array with shape (num_time_steps + 1, num_spatial_points)
    """
    num_spatial_points = len(u_initial)
    u_solution = np.zeros((num_time_steps + 1, num_spatial_points))
    u_solution[0, :] = u_initial
    u = u_initial.copy()

    def F(u_val):
        # Create differentiation matrix for first derivative (central difference)
        D = np.zeros((num_spatial_points, num_spatial_points))
        for i in range(num_spatial_points):
            if i > 0:
                D[i, i - 1] = -1
            if i < num_spatial_points - 1:
                D[i, i + 1] = 1
        
        D = D / (2 * h)
        
        # Apply advection equation: u_t = -c*u_x
        return -c * np.dot(D, u_val)

    for n in range(num_time_steps):
        k1 = F(u)
        k2 = F(u + (k / 2) * k1)
        k3 = F(u + (k / 2) * k2)
        k4 = F(u + k * k3)

        u = u + (k / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        u_solution[n + 1, :] = u

    return u_solution
