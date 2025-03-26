import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gibbsphen.solvers import wave_equation_rk4
from gibbsphen.solvers import advection_equation_solver
from gibbsphen.utils.visualization import make_animation
from gibbsphen.utils.initial_conditions import right_triangle_initial_condition, square_pulse_initial_condition

def run_gibbs_phenomenon_simulation(initial_condition_type="triangle", resolution=0.001):
    """
    Run a simulation demonstrating the Gibbs phenomenon in the wave equation.
    
    Parameters:
    -----------
    initial_condition_type : str
        Type of initial condition to use ("triangle", "square", or "sawtooth")
    resolution : float
        Spatial resolution (h) for the simulation
        
    Returns:
    --------
    u_solution : numpy.ndarray
        Solution array with shape (num_time_steps + 1, num_spatial_points)
    """
    # Simulation parameters
    spatial_domain_length = 1.0
    h = resolution
    num_spatial_points = int(spatial_domain_length / h)
    print(f"Spatial resolution: h = {h}, num spatial points: {num_spatial_points}")

    time_domain_length = 0.05

    c = 1.0  # Wave speed
    
    CFL = 0.2  # Courant-Friedrichs-Lewy number for stability
    k = CFL * h / c  # Time step size
    num_time_steps = int(time_domain_length / k)
    print(f"Time resolution: k = {k}, num time steps: {num_time_steps}")
    
    # Create spatial grid
    x = np.linspace(0, spatial_domain_length, num_spatial_points)

    # Create initial conditions based on selected type
    if initial_condition_type == "triangle":
        # Right triangle from (0,1) to (0.5,0)
        u_initial = right_triangle_initial_condition(x, base=0.5, height=1.0, position=0.0)
        condition_name = "triangle"
    elif initial_condition_type == "square":
        # Square pulse centered at x=0.5
        u_initial = square_pulse_initial_condition(x, width=0.2, height=1.0, center=0.5)
        condition_name = "square_pulse"
    elif initial_condition_type == "sawtooth":
        # Sawtooth wave (multiple triangles)
        u_initial = np.zeros_like(x)
        for i, xi in enumerate(x):
            # Create a sawtooth pattern with period 0.2
            rel_pos = (xi % 0.2) / 0.2
            if rel_pos < 0.5:
                u_initial[i] = 2.0 * rel_pos
            else:
                u_initial[i] = 2.0 * (1.0 - rel_pos)
        condition_name = "sawtooth"
    else:
        # Default to triangle
        u_initial = right_triangle_initial_condition(x, base=0.5, height=1.0, position=0.0)
        condition_name = "triangle"
    
    v_initial = np.zeros(num_spatial_points)  # Initial velocity is zero

    # Solve the wave equation
    # u_solution = wave_equation_rk4(u_initial, v_initial, c, h, k, num_time_steps)
    u_solution = advection_equation_solver(u_initial, c, h, k, num_time_steps)
    # Create animation
    output_dir = "gibbs_phenomenon_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"gibbs_phenomenon_{condition_name}_h_{h}.gif")
    
    make_animation(
        u_solution, 
        spatial_domain_length, 
        f"Gibbs Phenomenon: {condition_name.replace('_', ' ').title()} (h = {h}, c = {c})", 
        "Position (x)", "Amplitude (u)", 
        output_file
    )
    
    print(f"Animation saved to {output_file}")
    return u_solution

if __name__ == "__main__":
    # Run simulations with different initial conditions to demonstrate Gibbs phenomenon
    u_solution_triangle = run_gibbs_phenomenon_simulation("triangle")
    u_solution_square = run_gibbs_phenomenon_simulation("square")
    u_solution_sawtooth = run_gibbs_phenomenon_simulation("sawtooth")
    
    print("All simulations completed successfully.") 
    