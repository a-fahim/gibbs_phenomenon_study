import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    

def make_animation(u_solution, spatial_domain_length, title, xlabel, ylabel, filename):
    """
    Creates and saves an animation of the wave equation solution.
    
    Parameters:
    -----------
    u_solution : numpy.ndarray
        Solution array with shape (num_time_steps + 1, num_spatial_points)
    spatial_domain_length : float
        Length of the spatial domain
    title : str
        Title for the animation
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    filename : str
        Filename to save the animation (should end with .gif)
        
    Returns:
    --------
    ani : matplotlib.animation.FuncAnimation
        Animation object
    """
    num_time_steps = u_solution.shape[0] - 1
    num_spatial_points = u_solution.shape[1]
    x = np.linspace(0, spatial_domain_length, num_spatial_points)
    
    fig, ax = plt.subplots()
    line, = ax.plot(x, u_solution[0, :])
    ax.set_xlim(0, spatial_domain_length)
    ax.set_ylim(0.9 * np.min(u_solution), 1.1 * np.max(u_solution))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    def animate(i):
        line.set_ydata(u_solution[i, :])
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=num_time_steps + 1, interval=50, blit=True)

    # Save as GIF
    ani.save(filename, writer='pillow', fps=30) # save as a gif file.

    return ani 