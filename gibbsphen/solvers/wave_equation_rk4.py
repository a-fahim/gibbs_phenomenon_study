import numpy as np

def wave_equation_rk4(u_initial, v_initial, c, h, k, num_time_steps):
    """
    Solves the 1D wave equation using a combined finite-difference and Runge-Kutta (RK4) method.
    
    Parameters:
    -----------
    u_initial : numpy.ndarray
        Initial displacement values at spatial points
    v_initial : numpy.ndarray
        Initial velocity values at spatial points
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
    v = v_initial.copy()
    u = u_initial.copy()

    def F(y):
        u_val = y[:num_spatial_points]
        v_val = y[num_spatial_points:]

        A = np.zeros((num_spatial_points, num_spatial_points))
        for i in range(num_spatial_points):
            if i > 0:
                A[i, i - 1] = 1
            A[i, i] = -2
            if i < num_spatial_points - 1:
                A[i, i + 1] = 1

        A = A / (h**2)
        c2Au = c**2 * np.dot(A, u_val)
        return np.concatenate([v_val, c2Au])

    y = np.concatenate([u, v])

    for n in range(num_time_steps):
        k1 = F(y)
        k2 = F(y + (k / 2) * k1)
        k3 = F(y + (k / 2) * k2)
        k4 = F(y + k * k3)

        y = y + (k / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        u = y[:num_spatial_points]
        v = y[num_spatial_points:]
        u_solution[n + 1, :] = u

    return u_solution 