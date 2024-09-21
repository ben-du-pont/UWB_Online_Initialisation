import numpy as np
from scipy.optimize import minimize


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





class TrajectoryOptimization:


    def __init__(self, method="FIM", bounds=[(-float('inf'), float('inf')), (-float('inf'), float('inf')), (-float('inf'), float('inf'))], default_fim_noise_variance=0.2):

        self.fim_noise_variance = default_fim_noise_variance
        self.method = method  # Method for optimization (GDOP or FIM)
        self.bounds = bounds  # Bounds for the optimization

        self.anchor_estimate = None
        self.anchor_estimate_variance = None

    def compute_FIM(self, target_estimator, measurements, noise_variance, epsilon=1e-6):
        """Compute the Fisher Information Matrix (FIM) given a set of measurements corresponding to drone positions in space, and the target position, i.e the anchor to initialise
        The noise model is assumed to be zero-mean Gaussian with a distance dependent variance.

        Parameters:
        - target_estimator: list of floats, the estimated position of the target [x, y, z]
        - measurements: list of lists, the measurements corresponding to drone positions in space [[x1, y1, z1], [x2, y2, z2], ...]
        - noise_variance: float, the variance of the noise in the measurements
        - epsilon: float, a small value to avoid division by zero

        Returns:
        - FIM: numpy array, the Fisher Information Matrix (FIM) of the target position, 3x3 matrix
        """

        x0, y0, z0 = target_estimator
        measurements = np.array(measurements)
        noise_variance = self.fim_noise_variance # Default noise variance

        # Compute distances between measurements and target
        z_m = np.linalg.norm(measurements - np.array([x0, y0, z0]), axis=1)

        # Differences in x, y, z coordinates
        x_differences = x0 - measurements[:, 0]
        y_differences = y0 - measurements[:, 1]
        z_differences = z0 - measurements[:, 2]

        # Compute derivatives dzm_dx, dzm_dy, dzm_dz (no need for np.newaxis)
        dzm_dx = x_differences / z_m
        dzm_dy = y_differences / z_m
        dzm_dz = z_differences / z_m

        # Create the noise covariance matrix
        C_q = noise_variance * np.diag((1 + z_m)**2)
        
        # Invert C_q
        inv_C_q = np.linalg.inv(C_q)

        # Compute derivatives of C with respect to x, y, and z
        dC_dx = noise_variance * np.diag((1 + z_m) / z_m * x_differences)
        dC_dy = noise_variance * np.diag((1 + z_m) / z_m * y_differences)
        dC_dz = noise_variance * np.diag((1 + z_m) / z_m * z_differences)

        # Initialize Fisher Information Matrix (FIM)
        FIM = np.zeros((3, 3))

        # Compute the diagonal terms of FIM
        FIM[0, 0] = dzm_dx.T @ inv_C_q @ dzm_dx + 0.5 * np.trace(inv_C_q @ dC_dx @ inv_C_q @ dC_dx)
        FIM[1, 1] = dzm_dy.T @ inv_C_q @ dzm_dy + 0.5 * np.trace(inv_C_q @ dC_dy @ inv_C_q @ dC_dy)
        FIM[2, 2] = dzm_dz.T @ inv_C_q @ dzm_dz + 0.5 * np.trace(inv_C_q @ dC_dz @ inv_C_q @ dC_dz)

        # Compute the off-diagonal terms (cross-terms)
        FIM[0, 1] = dzm_dx.T @ inv_C_q @ dzm_dy + 0.5 * np.trace(inv_C_q @ dC_dx @ inv_C_q @ dC_dy)
        FIM[0, 2] = dzm_dx.T @ inv_C_q @ dzm_dz + 0.5 * np.trace(inv_C_q @ dC_dx @ inv_C_q @ dC_dz)
        FIM[1, 2] = dzm_dy.T @ inv_C_q @ dzm_dz + 0.5 * np.trace(inv_C_q @ dC_dy @ inv_C_q @ dC_dz)

        # Fill the symmetric terms
        FIM[1, 0] = FIM[0, 1]
        FIM[2, 0] = FIM[0, 2]
        FIM[2, 1] = FIM[1, 2]

        return FIM

    def compute_GDOP(self, target_coords, measurements):
        """Compute the Geometric Dilution of Precision (GDOP) given a set of measurements corresponding to drone positions in space, and the target position, i.e the anchor to initialise
        
        Parameters:
        - target_coords: list of floats, the estimated position of the target [x, y, z]
        - measurements: list of lists, the measurements corresponding to drone positions in space [[x1, y1, z1], [x2, y2, z2], ...]
        
        Returns:
        - gdop: float, the Geometric Dilution of Precision (GDOP) of the target position"""

        
        if len(measurements) < 4:
            return float('inf') # Not enough points to calculate GDOP
    
        x,y,z = target_coords
        A = []
        for measurement in measurements:
            x_i, y_i, z_i = measurement
            R = np.linalg.norm([x_i-x, y_i-y, z_i-z])
            A.append([(x_i-x)/R, (y_i-y)/R, (z_i-z)/R, 1])

        A = np.array(A)

        try:
            inv_at_a = np.linalg.inv(A.T @ A)
            gdop = np.sqrt(np.trace(inv_at_a))
            if gdop is not None:
                return gdop
            else:
                return float('inf')
        except np.linalg.LinAlgError:
            return float('inf')  # Matrix is singular, cannot compute GDOP
        
    def evaluate_gdop(self, new_measurements, anchor_position, previous_measurements):
        """
        Evaluate the GDOP using the previous and new measurements.
        """
        # Calculate the GDOP using the previous and new measurements
        new_measurements = np.array(new_measurements).reshape((-1, 3))
        gdop = self.compute_GDOP(anchor_position, np.vstack([previous_measurements, new_measurements]))
        return gdop

    def evaluate_fim(self, new_measurements, anchor_estimator, anchor_variance, previous_measurements):
        """
        Evaluate the FIM using the previous and new measurements.
        """
        # Calculate the FIM using the previous and new measurements
        
        new_measurements = np.array(new_measurements).reshape((-1, 3))
        anchor_position = anchor_estimator[:3]
        fim = self.compute_FIM(anchor_position, np.vstack([previous_measurements, new_measurements]), anchor_variance)
        fim_det_inverse = 1 / np.linalg.det(fim)
        return fim_det_inverse

    def cartesian_to_spherical(self, x, y, z, center):
        """Convert Cartesian coordinates to spherical coordinates
        Parameters:
        - x: float, the x-coordinate
        - y: float, the y-coordinate
        - z: float, the z-coordinate
        - center: list of floats, the center of the spherical coordinates [x_center, y_center, z_center]
        
        Returns:
        - r: float, the radial distance
        - theta: float, the polar angle
        - phi: float, the azimuthal angle
        """

        x -= center[0]
        y -= center[1]
        z -= center[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi
    
    def spherical_to_cartesian(self, r, theta, phi, center):
        """Convert spherical coordinates to Cartesian coordinates
        
        Parameters:
        - r: float, the radial distance
        - theta: float, the polar angle
        - phi: float, the azimuthal angle
        - center: list of floats, the center of the spherical coordinates [x_center, y_center, z_center]
        
        Returns:
        - x: float, the x-coordinate
        - y: float, the y-coordinate
        - z: float, the z-coordinate
        """

        x = r * np.sin(theta) * np.cos(phi) + center[0]
        y = r * np.sin(theta) * np.sin(phi) + center[1]
        z = r * np.cos(theta) + center[2]
        return x, y, z
        
    def get_spherical_bounds(self, center, radius):
        """
        Get the bounds for the local spherical coordinates given the center and radius.
        
        Parameters:
        - center: list of floats, the center of the spherical coordinates [x_center, y_center, z_center]
        - radius: float, the radius of the spherical coordinates
        
        Returns:
        - bounds_theta: tuple of floats, the bounds for the polar angle (theta) in radians
        - bounds_phi: tuple of floats, the bounds for the azimuthal angle (phi) in radians
        - center_in_bounds: bool, whether the center is inside the Cartesian bounds
        """

        # Get the Cartesian bounds (x_min, x_max), (y_min, y_max), (z_min, z_max)
        cartesian_bounds = self.bounds

        def intersect_phi_intervals(phi1_min, phi1_max, phi2_min, phi2_max):
            if phi1_min is None or phi2_min is None:
                return None, None

            # Find the intersection of two intervals
            phi_min = max(phi1_min, phi2_min)
            phi_max = min(phi1_max, phi2_max)

            if phi_min > phi_max:  # No valid intersection
                return None, None

            return phi_min, phi_max



        x_min, x_max = cartesian_bounds[0]
        y_min, y_max = cartesian_bounds[1]
        z_min, z_max = cartesian_bounds[2]
        x_c, y_c, z_c = center # Center of the spherical coordinates over which to optimize

        # Bounds for theta (from z-bounds)
        theta_min, theta_max = 0, np.pi  # Default bounds for theta

        if z_c > z_max:
            print("Optimisation not possible in the specified bounds, z_c > z_max")
            return (None, None), (None, None)
        if z_c < z_min:
            print("Optimisation not possible in the specified bounds, z_c < z_min")
            return (None, None), (None, None)

        # Here we apply a strict bounds strategy where we restrict the sphere by a half-sphere in directions that are outside the bounds
        if abs(z_c - z_max) < radius:
            theta_min = np.pi/2 # np.arccos((z_max - z_c) / radius)
        if abs(z_c - z_min) < radius:
            theta_max = np.pi/2 # np.arccos((z_min - z_c) / radius)

        # Default bounds for phi (entire circle)
        phi_min, phi_max = 0, 2 * np.pi  

        # x bounds check
        phi_min_x, phi_max_x = phi_min, phi_max  # Default for x
        if x_c > x_max:

            print("Optimisation not possible in the specified bounds, x_c > x_max")
            return (None, None), (None, None)
        if x_c < x_min:

            print("Optimisation not possible in the specified bounds, x_c < x_min")
            return (None, None), (None, None)



        if abs(x_c - x_max) < radius:
            phi_min_x = np.pi/2
            phi_max_x = 3*np.pi/2
        if abs(x_c - x_min) < radius:
            phi_min_x = -np.pi/2
            phi_max_x = np.pi/2

        # y bounds check
        phi_min_y, phi_max_y = phi_min, phi_max  # Default for y
        if y_c > y_max:
            print("Optimisation not possible in the specified bounds, y_c > y_max")
            return (None, None), (None, None)
        if y_c < y_min:

            print("Optimisation not possible in the specified bounds, y_c < y_min")
            return (None, None), (None, None)


        if abs(y_c - y_max) < radius:
            phi_min_y = np.pi
            phi_max_y = 2 * np.pi
        if abs(y_c - y_min) < radius:
            phi_min_y = 0
            phi_max_y = np.pi

        # Combine the intervals for x and y using intersection
        phi_min, phi_max = intersect_phi_intervals(phi_min_x, phi_max_x, phi_min_y, phi_max_y)



        return (theta_min, theta_max), (phi_min, phi_max)





    def optimize_waypoints_incrementally_spherical(self, drone_position, anchor_estimator, anchor_estimate_variance, previous_measurements, remaining_trajectory, radius_of_search = 1, max_waypoints=8, marginal_gain_threshold=0.01):
        anchor_estimate = anchor_estimator[:3]
        self.anchor_estimate = anchor_estimate
        self.anchor_estimate_variance = anchor_estimate_variance

        anchor_estimate_variance = [var for var in anchor_estimate_variance]

        best_waypoints = [drone_position] # Keep track of the best waypoints

        if self.method == "FIM":
            previous_fim_det = 1/np.linalg.det(self.compute_FIM(anchor_estimate, previous_measurements, anchor_estimate_variance))
        elif self.method == "GDOP":
            previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = drone_position # previous_measurements[-1]

        # Compute the directional vector between the last two measurements
        # vector = np.array(previous_measurements[-1]) - np.array(previous_measurements[-2])
        vector = np.array(anchor_estimate) - np.array(drone_position) 

        # Use this vector to determine the initial guess for the new measurement, to be in the same direction as this
        initial_guess = self.cartesian_to_spherical(vector[0], vector[1], vector[2], [0,0,0])[1:]

        # This was used to evaluate if the initial guess was correct and useful
        # self.initial_guess = [drone_position, self.spherical_to_cartesian(radius_of_search, initial_guess[0], initial_guess[1], drone_position)]

        for i in range(max_waypoints):

            # On the first iteration, use double the radius of search ??
            current_radius = radius_of_search * 1 if i == 0 else radius_of_search

            def objective(new_measurement):
                new_measurement = self.spherical_to_cartesian(current_radius, new_measurement[0], new_measurement[1], last_measurement)
                new_measurement = np.array(new_measurement).reshape(1, -1)

                if self.method == "GDOP":
                    gdop = self.evaluate_gdop(new_measurement, anchor_estimate, previous_measurements)
                    return gdop
                
                elif self.method == "FIM":
                    fim_gain = self.evaluate_fim(new_measurement, anchor_estimate, anchor_estimate_variance, previous_measurements)
                    return fim_gain  # Objective: Minimize inverse FIM determinant

            # Define bounds for new measurement within the specified bounds
            bounds = self.get_spherical_bounds(last_measurement, current_radius)

            bounds_theta = bounds[0]
            bounds_psi = bounds[1]


            if None in bounds_theta or None in bounds_psi:
                print(f"Optimisation not possible in the specified bounds. Stopping optimization.")
                break
            

            # Optimize for the next waypoint
            result = minimize(objective, initial_guess, method='Powell', bounds=bounds)

            if result.success:
                new_waypoint = result.x
                new_waypoint = self.spherical_to_cartesian(radius_of_search ,new_waypoint[0], new_waypoint[1], last_measurement)

                if self.method == "GDOP":
                    new_gdop = self.evaluate_gdop(new_waypoint, anchor_estimate, previous_measurements)

                elif self.method == "FIM":
                    new_fim_det = self.evaluate_fim(new_waypoint, anchor_estimator, anchor_estimate_variance, previous_measurements)

                # Check marginal gain
                if self.method == "GDOP":
                    gdop_gain = previous_gdop - new_gdop

                    if gdop_gain/previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain/previous_gdop}. Stopping optimization.")
                        break

                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det

                    if fim_gain/previous_fim_det < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {fim_gain/previous_fim_det}. Stopping optimization.")
                        break

                # Update best waypoints and metrics
                best_waypoints.append(new_waypoint)

                if np.linalg.norm(new_waypoint - anchor_estimate) < 2*radius_of_search:
                    print(f"Got close enough to the anchor. Stopping optimization.")
                    break

                previous_measurements = np.vstack([previous_measurements, new_waypoint])

                if self.method == "GDOP":
                    previous_gdop = new_gdop

                elif self.method == "FIM":
                    previous_fim_det = new_fim_det

                last_measurement = new_waypoint

            else:
                print(f"Optimization failed at waypoint {i + 1}")
                break

            # Initial guess for the next waypoint
            initial_guess = result.x
            # self.initial_guess.append(self.spherical_to_cartesian(radius_of_search, initial_guess[0], initial_guess[1], last_measurement))

        print(f"Optimization completed with {len(best_waypoints)} waypoints.")

        return best_waypoints
                

    def optimize_waypoints_incrementally_cubical(self, anchor_estimator, anchor_estimate_variance, previous_measurements, remaining_trajectory, bound_size=[5,5,5], max_waypoints=8, marginal_gain_threshold=0.01):
        anchor_estimate = anchor_estimator[:3]
        
        best_waypoints = []
        previous_fim_det = self.compute_FIM(anchor_estimate, anchor_estimate_variance, previous_measurements)
        previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = previous_measurements[-1]

        for i in range(max_waypoints):
            def objective(new_measurement):
                new_measurement = np.array(new_measurement).reshape(1, -1)
                if self.method == "GDOP":
                    gdop = self.evaluate_gdop(new_measurement, anchor_estimate, previous_measurements)
                    return gdop
                elif self.method == "FIM":
                    fim_gain = self.evaluate_fim(new_measurement, anchor_estimator, anchor_estimate_variance, previous_measurements)
                    return fim_gain  # Objective: Minimize inverse FIM determinant

            # Define bounds for new measurement within the specified bounds
            bounds = [
                (last_measurement[0] - bound_size[0]/2, last_measurement[0] + bound_size[0]/2),
                (last_measurement[1] - bound_size[1]/2, last_measurement[1] + bound_size[1]/2),
                (last_measurement[2] - bound_size[2]/2, last_measurement[2] + bound_size[2]/2)
            ]

            # Initial guess for the next waypoint
            initial_guess = last_measurement + np.random.uniform(-bound_size[0]/4, bound_size[0]/4, size=3)

            # Optimize for the next waypoint
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

            if result.success:
                new_waypoint = result.x
                if self.method == "GDOP":
                    new_gdop = self.evaluate_gdop(new_waypoint, anchor_estimate, previous_measurements)

                elif self.method == "FIM":
                    new_fim_det = self.evaluate_fim(new_waypoint, anchor_estimator, anchor_estimate_variance, previous_measurements)

                # Check marginal gain
                if self.method == "GDOP":
                    gdop_gain = previous_gdop - new_gdop
                    if gdop_gain/previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain/previous_gdop}. Stopping optimization.")
                        break
                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det

                    if fim_gain/previous_fim_det < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {fim_gain/previous_fim_det}. Stopping optimization.")
                        break

                # Update best waypoints and metrics
                best_waypoints.append(new_waypoint)
                previous_measurements = np.vstack([previous_measurements, new_waypoint])
                if self.method == "GDOP":
                    previous_gdop = new_gdop
                elif self.method == "FIM":
                    previous_fim_det = new_fim_det
                last_measurement = new_waypoint

            else:
                print(f"Optimization failed at waypoint {i + 1}")
                break

        return best_waypoints
    

    def optimize_return_waypoints_incrementally_spherical(self, drone_position, anchor_estimator, anchor_estimate_variance, previous_measurements, target_point, radius_of_search=1, max_waypoints=8, marginal_gain_threshold=0.01, lambda_penalty=10.0):
        

        anchor_estimate = anchor_estimator[:3]
        self.anchor_estimate = anchor_estimate
        self.anchor_estimate_variance = anchor_estimate_variance

        anchor_estimate_variance = [var for var in anchor_estimate_variance]

        best_waypoints = [] # Keep track of the best waypoints

        if self.method == "FIM":
            previous_fim_det = 1/np.linalg.det(self.compute_FIM(anchor_estimate, previous_measurements, anchor_estimate_variance))
        elif self.method == "GDOP":
            previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = drone_position # previous_measurements[-1]

        # Compute the directional vector between the last two measurements
        # vector = np.array(previous_measurements[-1]) - np.array(previous_measurements[-2])
        vector = np.array(anchor_estimate) - np.array(drone_position) 

        # Use this vector to determine the initial guess for the new measurement, to be in the same direction as this
        initial_guess = self.cartesian_to_spherical(vector[0], vector[1], vector[2], [0,0,0])[1:]

        # This was used to evaluate if the initial guess was correct and useful
        # self.initial_guess = [drone_position, self.spherical_to_cartesian(radius_of_search, initial_guess[0], initial_guess[1], drone_position)]

        for i in range(max_waypoints):

            # On the first iteration, use double the radius of search ??
            current_radius = radius_of_search * 1 if i == 0 else radius_of_search

            def objective(new_measurement):
                new_measurement = self.spherical_to_cartesian(current_radius, new_measurement[0], new_measurement[1], last_measurement)
                new_measurement = np.array(new_measurement).reshape(1, -1)

                if self.method == "GDOP":
                    gdop = self.evaluate_gdop(new_measurement, anchor_estimate, previous_measurements)
                    return gdop
                
                elif self.method == "FIM":
                    fim_gain = self.evaluate_fim(new_measurement, anchor_estimate, anchor_estimate_variance, previous_measurements)
                    return fim_gain  # Objective: Minimize inverse FIM determinant

            # Define bounds for new measurement within the specified bounds
            bounds = self.get_spherical_bounds(last_measurement, current_radius)

            bounds_theta = bounds[0]
            bounds_psi = bounds[1]


            if None in bounds_theta or None in bounds_psi:
                print(f"Optimisation not possible in the specified bounds. Stopping optimization.")
                break
            

            # Optimize for the next waypoint
            result = minimize(objective, initial_guess, method='Powell', bounds=bounds)

            if result.success:
                new_waypoint = result.x
                new_waypoint = self.spherical_to_cartesian(radius_of_search ,new_waypoint[0], new_waypoint[1], last_measurement)

                if self.method == "GDOP":
                    new_gdop = self.evaluate_gdop(new_waypoint, anchor_estimate, previous_measurements)

                elif self.method == "FIM":
                    new_fim_det = self.evaluate_fim(new_waypoint, anchor_estimator, anchor_estimate_variance, previous_measurements)

                # Check marginal gain
                if self.method == "GDOP":
                    gdop_gain = previous_gdop - new_gdop

                    if gdop_gain/previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain/previous_gdop}. Stopping optimization.")
                        break

                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det

                    if fim_gain/previous_fim_det < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {fim_gain/previous_fim_det}. Stopping optimization.")
                        break

                # Update best waypoints and metrics
                best_waypoints.append(new_waypoint)

                if np.linalg.norm(new_waypoint - anchor_estimate) < 2*radius_of_search:
                    print(f"Got close enough to the anchor. Stopping optimization.")
                    break

                previous_measurements = np.vstack([previous_measurements, new_waypoint])

                if self.method == "GDOP":
                    previous_gdop = new_gdop

                elif self.method == "FIM":
                    previous_fim_det = new_fim_det

                last_measurement = new_waypoint

            else:
                print(f"Optimization failed at waypoint {i + 1}")
                break

            # Initial guess for the next waypoint
            initial_guess = result.x

        best_optimal_waypoints = best_waypoints

        
        best_waypoints = []  # Track the best waypoints for the return path

        if self.method == "FIM":
            previous_fim_det = 1/np.linalg.det(self.compute_FIM(anchor_estimate, previous_measurements, anchor_estimate_variance))
        elif self.method == "GDOP":
            previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = best_optimal_waypoints[-1]  # Start from the current position

        # Set the initial guess for the next waypoint to be towards the target point

        vector_to_target = np.array(target_point) - np.array(best_optimal_waypoints[-1])
        initial_guess = self.cartesian_to_spherical(vector_to_target[0], vector_to_target[1], vector_to_target[2], [0, 0, 0])[1:]

        for i in range(max_waypoints):
            current_radius = radius_of_search * 1 if i == 0 else radius_of_search

            def objective(new_measurement):
                new_measurement = self.spherical_to_cartesian(current_radius, new_measurement[0], new_measurement[1], last_measurement)
                new_measurement = np.array(new_measurement).reshape(1, -1)
                
                # Compute FIM or GDOP gain for the new measurement
                if self.method == "GDOP":
                    gdop = self.evaluate_gdop(new_measurement, anchor_estimate, previous_measurements)
                    obj_value = gdop
                elif self.method == "FIM":
                    fim_gain = self.evaluate_fim(new_measurement, anchor_estimate, anchor_estimate_variance, previous_measurements)
                    obj_value = fim_gain  # Objective: Minimize inverse FIM determinant

                # Compute the distance penalty towards the target point (return path)
                distance_penalty = lambda_penalty * np.linalg.norm(new_measurement - target_point)**2

                # Combine the objective value with the distance penalty
                return obj_value + distance_penalty

            # Define bounds for new measurement within the search radius
            bounds = self.get_spherical_bounds(last_measurement, current_radius)

            bounds_theta = bounds[0]
            bounds_psi = bounds[1]

            if None in bounds_theta or None in bounds_psi:
                print(f"Optimization not possible in the specified bounds. Stopping optimization.")
                break

            # Optimize for the next waypoint
            result = minimize(objective, initial_guess, method='Powell', bounds=bounds)

            if result.success:
                new_waypoint = result.x
                new_waypoint = self.spherical_to_cartesian(radius_of_search, new_waypoint[0], new_waypoint[1], last_measurement)

                if self.method == "GDOP":
                    new_gdop = self.evaluate_gdop(new_waypoint, anchor_estimate, previous_measurements)

                elif self.method == "FIM":
                    new_fim_det = self.evaluate_fim(new_waypoint, anchor_estimator, anchor_estimate_variance, previous_measurements)

                # Check marginal gain and termination condition
                if self.method == "GDOP":
                    gdop_gain = previous_gdop - new_gdop

                    if gdop_gain / previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain / previous_gdop}. Stopping optimization.")
                        break

                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det

                    if fim_gain / previous_fim_det < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {fim_gain / previous_fim_det}. Stopping optimization.")
                        break

                # Check if we are close enough to the target point
                # if np.linalg.norm(new_waypoint - np.array(target_point)) < 2 * radius_of_search:
                #     print(f"Got close enough to the target point. Stopping optimization.")
                #     break

                # Update best waypoints and metrics
                best_waypoints.append(new_waypoint)
                previous_measurements = np.vstack([previous_measurements, new_waypoint])

                if self.method == "GDOP":
                    previous_gdop = new_gdop

                elif self.method == "FIM":
                    previous_fim_det = new_fim_det

                last_measurement = new_waypoint

            else:
                print(f"Optimization failed at waypoint {i + 1}")
                break

            # Update the initial guess for the next waypoint
            initial_guess = result.x

        print(f"Return path optimization completed with {len(best_waypoints)} waypoints.")
        return best_optimal_waypoints, best_waypoints

    def compute_new_mission_waypoints(self, initial_position, initial_remaining_waypoints, optimal_waypoints, method="return_to_closest", return_waypoints=None):
        """Compute the new mission waypoints based on the initial position, initial waypoints, and optimal waypoints
        
        Parameters:
        - initial_position: list of floats, the initial position of the drone [x, y, z]
        - initial_waypoints: list of lists, the initial waypoints to follow
        - optimal_waypoints: list of lists, the optimal waypoints to follow
        - method: str, the method to use to compute the new mission waypoints
        
        Returns:
        - new_mission_waypoints: list of lists, the new mission waypoints to follow
        """
        
        # Note: initial position is actually the previous passed waypoint
        new_mission_waypoints = []
        
        def closest_point_on_line(A, B, C):
            # Convert points to numpy arrays
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)
            
            # Vector AB and AC
            AB = B - A
            AC = C - A
            
            # Project AC onto AB
            t = np.dot(AC, AB) / np.dot(AB, AB)
            
            # Find the closest point P on the line
            P = A + t * AB

            if t < 0:
                return A
            if t > 1:
                return B
            
            return P

        def distance_on_segment(A, B, C, d):
            # Convert points to numpy arrays for vector operations
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)
            
            # Compute vector CB
            CB = B - C
            distance_CB = np.linalg.norm(CB)  # Distance from C to B
            
            if d >= distance_CB:
                # If the distance d is greater than or equal to distance CB, return B
                return B
            
            # Otherwise, find the direction vector from C to B (unit vector)
            CB_unit = CB / distance_CB
            
            # Compute the point P on the segment at distance d from C towards B
            P = C + d * CB_unit
            
            return P

        if not optimal_waypoints:
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = []
            return new_mission_waypoints, optimal_waypoints, link_waypoints, initial_remaining_waypoints
        




        if method == "strict_return":
            closest_point = closest_point_on_line(initial_position, initial_remaining_waypoints[0], optimal_waypoints[-1])
            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.append(closest_point)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = []
            optimal_waypoints.append(closest_point)

        if method == "return_to_initial":
            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.append(initial_position)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = [initial_position]

        if method == "straight_to_wapoint":
            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            if initial_remaining_waypoints:
                link_waypoints = [initial_remaining_waypoints[0]]


        if method == "return_to_closest":
            closest_point = closest_point_on_line(initial_position, initial_remaining_waypoints[0], optimal_waypoints[-1])
            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.append(closest_point)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = [closest_point]

        if method == "hybrid_return":
            closest_point = closest_point_on_line(initial_position, initial_remaining_waypoints[0], optimal_waypoints[-1])
            distance = np.tan(np.pi/3)*np.linalg.norm(closest_point - optimal_waypoints[-1])
            hybrid_point = distance_on_segment(initial_position, initial_remaining_waypoints[0], closest_point, distance)

            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.append(hybrid_point)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = []
            optimal_waypoints.append(closest_point)
        
        if method == "optimal":
            new_mission_waypoints.extend(optimal_waypoints)
            link_waypoints = return_waypoints
            new_mission_waypoints.extend(link_waypoints)
            new_mission_waypoints.extend(initial_remaining_waypoints)

        return new_mission_waypoints, optimal_waypoints, link_waypoints, initial_remaining_waypoints

def main():
    # Create an instance of the TrajectoryOptimization class
    trajectory_optimization = TrajectoryOptimization()

    def load_measurement_data_from_csv(path):

        def convert_str_to_list(s):
            # Replace 'nan', 'inf', and '-inf' with their corresponding numpy constants
            s = s.replace('nan', 'np.nan').replace('inf', 'np.inf').replace('-inf', '-np.inf')
            # Evaluate the string as a Python expression and return the result
            try:
                return eval(s)
            except Exception as e:
                # If evaluation fails, return the original string
                return s
    
        data = pd.read_csv(path, header=None, converters={i: convert_str_to_list for i in range(3)})
        data.columns = ['anchor_position_gt','measured_positions', 'measured_ranges']

        return data
    
    package_path = Path(__file__).parent.resolve()
    csv_dir = package_path / 'csv_files'

    df = load_measurement_data_from_csv(csv_dir / 'measurements.csv')
    
    csv_index = 0
    previous_measurements = df.iloc[csv_index]['measured_positions']
    anchor_gt = df.iloc[csv_index]['anchor_position_gt']

    half_length = len(previous_measurements) * 5 // 6
    previous_measurements = previous_measurements[:half_length]

    indices = np.random.choice(len(previous_measurements), size=1, replace=False)
    indices.sort()
    
    # Sample measurements
    previous_measurements = np.array(previous_measurements)
    sampled_measurements = previous_measurements[indices]
    sampled_measurements = sampled_measurements.tolist()

    sampled_measurements = previous_measurements.tolist()
    #waypoints = trajectory_optimization.optimize_waypoints_incrementally_cubical(anchor_estimate=anchor_gt, previous_measurements=sampled_measurements, remaining_trajectory=[], bound_size=[3,3,3], max_waypoints=20, marginal_gain_threshold=0.01)
    waypoints = trajectory_optimization.optimize_waypoints_incrementally_spherical(anchor_estimate=anchor_gt, previous_measurements=sampled_measurements, remaining_trajectory=[], radius_of_search=2, max_waypoints=20, marginal_gain_threshold=0.05)

    
     # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Previous Measurements
    prev_meas = np.array(sampled_measurements)
    ax.scatter(prev_meas[:,0], prev_meas[:,1], prev_meas[:,2], color='blue', label='Previous Measurements')

    # Anchor Ground Truth
    ax.scatter(anchor_gt[0], anchor_gt[1], anchor_gt[2], color='red', s=100, label='Anchor Ground Truth')

    # Waypoints
    if waypoints:
        waypoints_arr = np.array(waypoints)
        ax.scatter(waypoints_arr[:,0], waypoints_arr[:,1], waypoints_arr[:,2], color='green', label='Additional Measurements')
    guess = trajectory_optimization.initial_guess
    guess = np.array(guess)

    ax.scatter(guess[1:, 0], guess[1:, 1], guess[1:, 2], color='black', label='Initial Guess')

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()



    # INPUT: 
    # array of positions used to estimate the position until now
    # estimated position of the anchor
    # remaining trajectory of the drone as a discretised array of positions (this is not the waypoints but directly the discretised path, can be used to calculate a deviation metric for example, or something else)
    
    
    # TO IMPLEMENT:
    # Objective function (given a path, output a metric) FIM ? GDOP ? COVARIANCE ?
    # Constraints - A cubic volume ?
    # Optimisation approach ? PSO, RRT, Genetic, SQP, 

    # Basically:
    # 1. Generate feasible paths
    # 2. Evaluate path wrt to the objective function (Optimality of the measurements but also path length)
    # 3. Select the best path

if __name__ == '__main__':
    main()