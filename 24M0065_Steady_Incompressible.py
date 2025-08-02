import numpy as np
import matplotlib.pyplot as plt

# Domain of the chamber
Lx = 3.0  # Length of the chamber in x-direction (m)
Ly = 4.0  # Length of the chamber in y-direction (m)
dx = 0.1  # Grid spacing in x-direction (m)
dy = 0.1  # Grid spacing in y-direction (m)
IM = int(Lx / dx) + 1  # Maximum number of grid points in x-direction
JM = int(Ly / dy) + 1  # Maximum number of grid points in y-direction
max_iter = 10000  # Maximum number of iterations provided for sulution
tolerance = 1e-4  # Convergence tolerance as stated in the problem statement

# The three boundary conditions provided in the problem statement are stored in a python dictionary
boundary_conditions = [
    {"psi1": 100, "psi2": 150, "psi3": 300},  # Test 1
    {"psi1": 100, "psi2": 200, "psi3": 300},  # Test 2
    {"psi1": 100, "psi2": 250, "psi3": 300}   # Test 3
]
# Initial guesses are stored as a python list
initial_guesses = [100.0, 150.0, 200.0]

def L2_norm(arr):
    """ This function will compute L2 norm for interior points only (i=2 to IM-1, j=2 to JM-1)
    In Python, indices start from 0, so i=2 corresponds to index 1, and IM-1 corresponds to index -2
    """
    return np.sqrt(np.sum(arr[1:-1, 1:-1]**2))

# Point Jacobi iterative method: This function will perform calculation till the solution converges
def point_jacobi(psi, max_iter, tolerance):
    psi_new = np.copy(psi)
    error_history = []  # The computed  errors are stored in list for convergence plot
    for iteration in range(max_iter):
        for i in range(1, IM - 1):
            for j in range(1, JM - 1):
                if not (i == int(1.5 / dx) and (0 <= j <= int(1.9 / dy) or j >= int(2.0 / dy))):  # This provide condtion Skip internal boundary condtion.
                    psi_new[i, j] = 0.25 * (psi[i+1, j] + psi[i-1, j] + psi[i, j+1] + psi[i, j-1])
        
        # Compute the error L2 norm for interior points
        error = L2_norm(psi_new - psi) / L2_norm(psi_new)
        error_history.append(error)  # The error values are stored in the error_history list
        
        #  The convergence condion is being checked here to check for final solution
        if error < tolerance:
            print(f"Converged after {iteration + 1} iterations with error = {error}")
            break
        
        # After each iteration the stream function will be updated for futher caclulations.
        psi = np.copy(psi_new)

    return psi, error, error_history  
    """ Once the solution converges, this function will return the final values of stream function, 
    final error and error history for each iteration"""

def run_solver(initial_guess, psi1, psi2, psi3):
    """" The stream function array will be initilized based on three guess values providedd 
    the problem statement"""
    psi = np.full((IM, JM), initial_guess)
    
    """ The left right, and top walls of the chamber are assiged with psi3  using slicing method.
    The walls represents line of constant stream function """
    psi[0, :] = psi3  # Left boundary
    psi[-1, :] = psi3  # Right boundary
    psi[:, -1] = psi3  # Top boundary

    # The bottom boundary consists of three section as describeed in the code.
    psi[:int(1.1/dx), 0] = psi3  # Left section
    psi[int(1.1/dx):int(2.0/dx), 0] = psi1  # Inlet
    psi[int(2.0/dx):, 0] = psi3  # Right section

    # The internal boundary start at x = 1.5 as shown figure in problem statement
    interior_x = int(1.5 / dx)
    psi[interior_x, :int(1.1/dy)] = psi1  # Lower part
    psi[interior_x, int(1.1/dy):int(2.0/dy)] = psi2  # Middle part
    psi[interior_x, int(2.0/dy):] = psi3  # Upper part
    
    # The point jacobi method is run with provided input as stream function, maximum iteration and tolerance value
    psi_converged, final_error, error_history = point_jacobi(psi, max_iter, tolerance)
    
    return psi_converged, final_error, error_history

# An empty list is created here in order to store the converged solutions.
results = []

for bc in boundary_conditions:
    """ This for loop will run the solver for each combination of initial guess
    and boundary condition """
    for guess in initial_guesses:
        print(f"Running solver with initial guess = {guess} and boundary conditions: {bc}")
        psi_converged, final_error, error_history = run_solver(guess, bc["psi1"], bc["psi2"], bc["psi3"])
        results.append((guess, bc, psi_converged, final_error, error_history))

# The converged solution at specific x-locations are printed for a given guess value, boundary condition 
x_locations = [0.0, 1.0, 2.0, 3.0]
for guess, bc, psi_converged, _, _ in results:
    print(f"\nConverged solution for initial guess = {guess} and boundary conditions: {bc}:")
    for x in x_locations:
        idx = int(x / dx)
        print(f"At x = {x} m:")
        print(psi_converged[idx, :])

# This part will generate contour plots for each combination of initial guess and boundary conditions
x = np.linspace(0, Lx, IM)
y = np.linspace(0, Ly, JM)
X, Y = np.meshgrid(x, y)

for guess, bc, psi_converged, _, _ in results:
    plt.figure()
    plt.contourf(X, Y, psi_converged.T, levels=50, cmap='viridis')
    plt.colorbar(label='Stream-function (ψ)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Streamline Pattern (Initial Guess = {guess}, BCs: {bc})')
    plt.show()

# Velocity field calculation and streamplot
for guess, bc, psi_converged, _, _ in results:
    # The grid points are defined here for plotting
    Grid_Points_along_x = np.linspace(0, Lx, IM)  # x-grid points
    Grid_Points_along_y = np.linspace(0, Ly, JM)  # y-grid points

    # Compute velocity field from stream function
    U = np.zeros((IM, JM))  # u = dψ/dy
    V = np.zeros((IM, JM))  # v = -dψ/dx

    # Central differencing is used here for velocity components computed from stream fucntion
    U[1:-1, 1:-1] = (psi_converged[1:-1, 2:] - psi_converged[1:-1, :-2]) / (2 * dy)  # u = dψ/dy
    V[1:-1, 1:-1] = -(psi_converged[2:, 1:-1] - psi_converged[:-2, 1:-1]) / (2 * dx)  # v = -dψ/dx

    # A mesh grid has been created here
    X, Y = np.meshgrid(Grid_Points_along_x, Grid_Points_along_y, indexing='xy')

    # The lines of codes below will plot the streamplot of the velocity field
    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, U.T, V.T, color='black', linewidth=1, density=1.5, arrowsize=1.5)  #  density and arrowsize can be adjusted by changing the numbers
    plt.title(f"Velocity Field Streamplot (Initial Guess = {guess}, BCs: {bc})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

# All the  convergence graphs for all cases  will be plotted in a single graph
plt.figure(figsize=(10, 6))  # It is  a single figure for all convergence histories
for guess, bc, _, _, error_history in results:
    plt.plot(error_history, label=f'Initial Guess = {guess}, BCs: {bc}')  # Each cases will be plotted here

# Adding  labels, title, and legend here
plt.title('Convergence History for All Cases')
plt.xlabel('Iteration')
plt.ylabel('Error (L2 Norm)')
plt.yscale('log')  # logarithmic scale has been used here  for better visualization
plt.legend()  # An specific legend is added to differentiate cases
plt.grid()  # Grid has aslo been to differentiate  for better readability
plt.show()