import numpy as np

def right_triangle_initial_condition(x, base=0.5, height=1.0, position=-0.5):
    """
    Creates a right triangle initial condition.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Spatial grid points
    base : float
        Base length of the triangle
    height : float
        Height of the triangle
    position : float
        Position of the left corner of the triangle
        
    Returns:
    --------
    u_initial : numpy.ndarray
        Initial condition values at spatial points
    """
    u_initial = np.zeros_like(x)
    
    # Calculate the slope of the hypotenuse
    slope = -height / base
    
    # Calculate the right edge of the triangle
    right_edge = position + base
    
    # Set values for points within the triangle
    for i, xi in enumerate(x):
        if position <= xi <= right_edge:
            # Linear function from (position, height) to (right_edge, 0)
            u_initial[i] = height + slope * (xi - position)
    
    return u_initial

def square_pulse_initial_condition(x, width=0.2, height=1.0, center=0.5):
    """
    Creates a square pulse initial condition.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Spatial grid points
    width : float
        Width of the pulse
    height : float
        Height of the pulse
    center : float
        Center position of the pulse
        
    Returns:
    --------
    u_initial : numpy.ndarray
        Initial condition values at spatial points
    """
    u_initial = np.zeros_like(x)
    
    # Calculate the left and right edges of the pulse
    left_edge = center - width / 2.0
    right_edge = center + width / 2.0
    
    # Set values for points within the pulse
    for i, xi in enumerate(x):
        if left_edge <= xi <= right_edge:
            u_initial[i] = height
    
    return u_initial 