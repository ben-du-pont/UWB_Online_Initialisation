import numpy as np
from scipy.optimize import minimize



class TrajectoryOptimization:


    def __init__(self):
        self.fim_noise_variance = 0.2  # Noise variance for the FIM calculation
        self.method = "FIM"  # Method to use for optimization (FIM or GDOP)
        self.initial_guess = [0, 0, 0]  # Initial guess for the optimization algorithm

    def calculate_fim(self, target_estimator, measurements, noise_variance):
        """
        Computes the Fisher Information Matrix (FIM) for the parameters (x0, y0, z0, gamma, beta)
        given the range measurements and noise characteristics.

        Parameters:
        x0, y0, z0: Coordinates of the object to estimate.
        gamma: Constant bias in the range measurements.
        beta: Linear bias in the range measurements.
        measurements: List of tuples [(x, y, z, r), ...] where (x, y, z) are the coordinates of known points
                    and r is the range measurement to the object.
        sigma_squared: The base variance of the noise in the measurements.

        Returns:
        FIM: Fisher Information Matrix (5x5 numpy array)
        """

        FIM = np.zeros((5, 5))

        var_x, var_y, var_z = noise_variance
        x0, y0, z0, gamma, beta = target_estimator

        # for measurement in measurements:
        #     xi, yi, zi = measurement
            
        #     # Calculate the true distance
        #     di = np.sqrt((xi - x0)**2 + (yi - y0)**2 + (zi - z0)**2)
            
        #     # Calculate the distance-dependent variance
        #     Cxi = var_x * (1 + di)**2
        #     Cyi = var_y * (1 + di)**2
        #     Czi = var_z * (1 + di)**2
            
        #     # Partial derivatives
        #     d_di_dx0 = (x0 - xi) / di
        #     d_di_dy0 = (y0 - yi) / di
        #     d_di_dz0 = (z0 - zi) / di   
            
        #     # Calculate the effective variance
        #     Ci = beta**2 * (d_di_dx0**2 / Cxi + d_di_dy0**2 / Cyi + d_di_dz0**2 / Czi)
            
        #     # Jacobian vector
        #     J = np.array([
        #         [-beta * d_di_dx0 / Cxi, -beta * d_di_dy0 / Cyi, -beta * d_di_dz0 / Czi, -1 / Ci, -di / Ci]
        #     ]).T
            
        #     # Fisher Information contribution from this measurement
        #     FIM_contrib = (1 / Ci) * np.dot(J, J.T)
            
        #     # Accumulate the Fisher Information Matrix
        #     FIM += FIM_contrib
        
        # # OPTION 2
        # FIM = np.zeros((3, 3))
    
        # for measurement in measurements:
        #     xi, yi, zi = measurement
            
        #     # Calculate the true distance
        #     di = np.sqrt((xi - x0)**2 + (yi - y0)**2 + (zi - z0)**2)
            
        #     # Partial derivatives for the distance with respect to x0, y0, z0
        #     d_di_dx0 = (x0 - xi) / di
        #     d_di_dy0 = (y0 - yi) / di
        #     d_di_dz0 = (z0 - zi) / di
            
        #     # Fisher Information contributions from this measurement
        #     FIM[0, 0] += (d_di_dx0**2) / var_x
        #     FIM[1, 1] += (d_di_dy0**2) / var_y
        #     FIM[2, 2] += (d_di_dz0**2) / var_z
        #     FIM[0, 1] += (d_di_dx0 * d_di_dy0) / np.sqrt(var_x * var_y)
        #     FIM[0, 2] += (d_di_dx0 * d_di_dz0) / np.sqrt(var_x * var_z)
        #     FIM[1, 2] += (d_di_dy0 * d_di_dz0) / np.sqrt(var_y * var_z)
        
        # # Since FIM is symmetric, we mirror the off-diagonal terms
        # FIM[1, 0] = FIM[0, 1]
        # FIM[2, 0] = FIM[0, 2]
        # FIM[2, 1] = FIM[1, 2]
        
        FIM = np.zeros((3, 3))  # Initialize the FIM matrix
        
        for measurement in measurements:
            x_i, y_i, z_i = measurement
            d_i = np.sqrt((x0 - x_i)**2 + (y0 - y_i)**2 + (z0 - z_i)**2)
            
            if d_i == 0:
                continue  # Avoid division by zero

            # Jacobian of the distance with respect to the anchor position
            jacobian = (np.array([x_i, y_i, z_i]) - np.array([x0, y0, z0])) / d_i

            # Update FIM
            FIM += (1 / (self.fim_noise_variance * (1 + d_i**2))) * np.outer(jacobian, jacobian)

        return FIM


    
    def compute_GDOP(self, target_coords, measurements):
        """Compute the Geometric Dilution of Precision (GDOP) given a set of measurements corresponding to drone positions in space"""
        
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

    def evaluate_fim(self, new_measurements, anchor_position, anchor_variance, previous_measurements):
        """
        Evaluate the FIM using the previous and new measurements.
        """
        # Calculate the FIM using the previous and new measurements
        
        new_measurements = np.array(new_measurements).reshape((-1, 3))

        fim = self.calculate_fim(anchor_position, np.vstack([previous_measurements, new_measurements]), anchor_variance)
        fim_det_inverse = 1 / np.linalg.det(fim)
        return fim_det_inverse

    def cartesian_to_spherical(self, x, y, z, center):
        """Convert Cartesian coordinates to spherical coordinates"""
        x -= center[0]
        y -= center[1]
        z -= center[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi
    
    def spherical_to_cartesian(self, r, theta, phi, center):
        """Convert spherical coordinates to Cartesian coordinates"""
        x = r * np.sin(theta) * np.cos(phi) + center[0]
        y = r * np.sin(theta) * np.sin(phi) + center[1]
        z = r * np.cos(theta) + center[2]
        return x, y, z
        
    def optimize_waypoints_incrementally_spherical(self, anchor_estimator, anchor_estimate_variance, previous_measurements, remaining_trajectory, radius_of_search = 1, max_waypoints=8, marginal_gain_threshold=0.01):
        anchor_estimate = anchor_estimator[:3]
        
        anchor_estimate_variance = [var for var in anchor_estimate_variance]
        anchor_estimate_variance = [0.5 for var in anchor_estimate_variance]
        best_waypoints = []
        previous_fim_det = 1/np.linalg.det(self.calculate_fim(anchor_estimator, previous_measurements, anchor_estimate_variance))
        previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = previous_measurements[-1]
        # Compute the directional vector between the last two measurements
        vector = np.array(previous_measurements[-1]) - np.array(previous_measurements[-2])
        # Use this vector to determine the initial guess for the new measurement, to be in the same direction as this
        initial_guess = self.cartesian_to_spherical(vector[0], vector[1], vector[2], [0,0,0])[1:]

        self.initial_guess = [previous_measurements[-1], self.spherical_to_cartesian(radius_of_search, initial_guess[0], initial_guess[1], previous_measurements[-1])]

        for i in range(max_waypoints):

            # On the first iteration, use double the radius of search
            current_radius = radius_of_search * 1 if i == 0 else radius_of_search

            def objective(new_measurement):
                new_measurement = self.spherical_to_cartesian(current_radius, new_measurement[0], new_measurement[1], last_measurement)
                new_measurement = np.array(new_measurement).reshape(1, -1)

                if self.method == "GDOP":
                    gdop = self.evaluate_gdop(new_measurement, anchor_estimate, previous_measurements)
                    return gdop
                elif self.method == "FIM":
                    fim_gain = self.evaluate_fim(new_measurement, anchor_estimator, anchor_estimate_variance, previous_measurements)
                    print(f"FIM det: {fim_gain}")
                    return fim_gain  # Objective: Minimize inverse FIM determinant

            # Define bounds for new measurement within the specified bounds
            bounds = [
                (0, np.pi),
                (0, 2*np.pi)
            ]

            

            # Optimize for the next waypoint
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

            if result.success:
                new_waypoint = result.x
                new_waypoint = self.spherical_to_cartesian(current_radius ,new_waypoint[0], new_waypoint[1], last_measurement)

                if self.method == "GDOP":
                    new_gdop = self.evaluate_gdop(new_waypoint, anchor_estimate, previous_measurements)

                elif self.method == "FIM":
                    new_fim_det = self.evaluate_fim(new_waypoint, anchor_estimator, anchor_estimate_variance, previous_measurements)

                # Check marginal gain
                if self.method == "GDOP":
                    gdop_gain = previous_gdop - new_gdop
                    print(f"NEW GDOP: {new_gdop}")
                    print(f"Marginal gain: {gdop_gain/previous_gdop}")
                    if gdop_gain/previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain/previous_gdop}. Stopping optimization.")
                        break
                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det
                    print(f"NEW FIM det: {new_fim_det}")
                    print(f"Marginal gain: {fim_gain/previous_fim_det}")
                    if fim_gain/previous_fim_det < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {fim_gain/previous_fim_det}. Stopping optimization.")
                        print(f"Number of waypoints: {len(best_waypoints)}")
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

            # Initial guess for the next waypoint
            initial_guess = result.x
            self.initial_guess.append(self.spherical_to_cartesian(radius_of_search, initial_guess[0], initial_guess[1], last_measurement))

        return best_waypoints
                

    def optimize_waypoints_incrementally_cubical(self, anchor_estimator, anchor_estimate_variance, previous_measurements, remaining_trajectory, bound_size=[5,5,5], max_waypoints=8, marginal_gain_threshold=0.01):
        anchor_estimate = anchor_estimator[:3]
        
        best_waypoints = []
        previous_fim_det = self.calculate_fim(anchor_estimate, anchor_estimate_variance, previous_measurements)
        previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = previous_measurements[-1]

        for _ in range(max_waypoints):
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
                    print(f"NEW GDOP: {new_gdop}")
                    print(f"Marginal gain: {gdop_gain/previous_gdop}")
                    if gdop_gain/previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain/previous_gdop}. Stopping optimization.")
                        break
                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det
                    print(f"NEW FIM det: {new_fim_det}")
                    print(f"Marginal gain: {fim_gain/previous_fim_det}")
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
                print(f"Optimization failed at waypoint {_ + 1}")
                break

        return best_waypoints
    

    # def create_optimal_trajectory(self, initial_position, optimal_points):
    #     """Create the optimal trajectory for the drone to follow given the initial position and the waypoints
        
    #     Parameters:
    #     - initial_position: list of floats, the initial position of the drone [x, y, z]
    #     - waypoints: list of lists, the waypoints to follow
        
    #     Returns:
    #     - spline_x: list of floats, the x component of the spline
    #     - spline_y: list of floats, the y component of the spline
    #     - spline_z: list of floats, the z component of the spline
    #     """

    #     if len(optimal_points) < 2:
    #         return Trajectory()
        
    #     trajectory = Trajectory()
    #     waypoints = []
    #     waypoints.append(Waypoint(initial_position[0], initial_position[1], initial_position[2]))

    #     for waypoint in optimal_points:
    #         waypoints.append(Waypoint(waypoint[0], waypoint[1], waypoint[2]))

    #     trajectory.construct_trajectory_spline(waypoints)

    #     return trajectory
    
    # def create_link_from_deviation_to_initial_trajectory(self, initial_trajectory, optimal_trajectory_points):
        
    #     if len(optimal_trajectory_points) < 2:
    #         return initial_trajectory
        
    #     trajectory = Trajectory()
    #     optimal_trajectory_waypoints = []
    #     for point in optimal_trajectory_points:
            
    #         optimal_trajectory_waypoints.append(Waypoint(point[0], point[1], point[2]))

    #     initial_position = optimal_trajectory_waypoints[0].x, optimal_trajectory_waypoints[0].y, optimal_trajectory_waypoints[0].z

    #     def find_closest_point_index(trajectory, point):

    #         distances = np.sqrt((trajectory.spline_x - point[0])**2 + 
    #                             (trajectory.spline_y - point[1])**2 + 
    #                             (trajectory.spline_z - point[2])**2)
    #         return np.argmin(distances)

    #     closest_index = find_closest_point_index(initial_trajectory, initial_position)

    #     cut_initial_trajectory = initial_trajectory
    #     cut_initial_trajectory.spline_x = initial_trajectory.spline_x[closest_index:]
    #     cut_initial_trajectory.spline_y = initial_trajectory.spline_y[closest_index:]
    #     cut_initial_trajectory.spline_z = initial_trajectory.spline_z[closest_index:]

    #     final_waypoint = optimal_trajectory_waypoints[-1]

    #     closest_index = find_closest_point_index(cut_initial_trajectory, [final_waypoint.x, final_waypoint.y, final_waypoint.z])
    #     closest_point = Waypoint(cut_initial_trajectory.spline_x[closest_index], cut_initial_trajectory.spline_y[closest_index], cut_initial_trajectory.spline_z[closest_index])

    #     cut_final_trajectory = cut_initial_trajectory
    #     cut_final_trajectory.spline_x = cut_initial_trajectory.spline_x[closest_index:]
    #     cut_final_trajectory.spline_y = cut_initial_trajectory.spline_y[closest_index:]
    #     cut_final_trajectory.spline_z = cut_initial_trajectory.spline_z[closest_index:]

    #     trajectory.construct_trajectory_spline([final_waypoint, closest_point])

    #     trajectory.spline_x = np.concatenate((trajectory.spline_x, cut_final_trajectory.spline_x,))
    #     trajectory.spline_y = np.concatenate((trajectory.spline_y, cut_final_trajectory.spline_y,))
    #     trajectory.spline_z = np.concatenate((trajectory.spline_z, cut_final_trajectory.spline_z,))

    #     return trajectory

    # def create_full_optimised_trajectory(self, initial_position, optimal_points, initial_trajectory):

    #     optimal_trajectory = self.create_optimal_trajectory(initial_position, optimal_points)
    #     link_trajectory = self.create_link_from_deviation_to_initial_trajectory(initial_trajectory, optimal_points)

    #     trajectory = Trajectory()
    #     trajectory.spline_x = np.concatenate((optimal_trajectory.spline_x, link_trajectory.spline_x))
    #     trajectory.spline_y = np.concatenate((optimal_trajectory.spline_y, link_trajectory.spline_y))
    #     trajectory.spline_z = np.concatenate((optimal_trajectory.spline_z, link_trajectory.spline_z))

    #     return trajectory

