import numpy as np
from scipy.optimize import least_squares, minimize
# from sklearn.mixture import GaussianMixture

# from sklearn.cluster import KMeans


from uwb_online_initialisation.trajectory_optimisation import TrajectoryOptimization

from copy import deepcopy
import csv


class UwbOnlineInitialisation:

    def __init__(self):
        
        self.drone_position = [] # To keep track of the current position of the drone in global frame

        # Default structure of the above dictionnary for one single anchor
        self.default_anchor_structure = {

            # Anchor initialisation status
            "status": "unseen", # Will be updated throughout the pipeline to keep track of the anchor status

            # Measurements from the initial rough estimation
            "distances_pre_rough_estimate": [], # Store the range measurements from the anchor to the drone before the trajectory optimisation
            "positions_pre_rough_estimate": [], # Store the positions of the drone when the range measurements were taken

            # Measurements from the optimal trajectory
            "distances_post_rough_estimate": [], # Store the range measurements from the anchor to the drone after the trajectory optimisation
            "positions_post_rough_estimate": [], # Store the positions of the drone when the range measurements were taken

            # Rough estimator
            "estimator_rough_linear": [0.0, 0.0, 0.0, 0.0, 1.0], # Store the rough estimate of the anchor position and bias terms for the linear least squares
            "covariance_matrix_rough_linear": [], # Store the covariance matrix of the rough estimate for the linear least squares
            "linear_ls_weights": [], # Store the weights of the linear least squares problem
            "linear_ls_outlier_counter": [], # Counter to store the number of consecutive times a measurement was flagged as an outlier

            
            # Refined non linear rough estimator
            "estimator_rough_non_linear": [0.0, 0.0, 0.0, 0.0, 0.0], # Store the rough estimate of the anchor position and bias terms for the non linear least squares
            "covariance_matrix_rough_non_linear": [], # Store the covariance matrix of the rough estimate for the non linear least squares

            # Final estimator
            "estimator": [0.0, 0.0, 0.0, 0.0, 0.0], # Store the final estimate of the anchor position and bias terms (Will constantly be updated and corresponds to the current estimate)
            "covariance_matrix": [], # Store the covariance matrix of the final estimate

            # Variables for the thresholding - are appended at every iteration
            "FIM": [], # Store the Fisher Information Matrix of the linear least squares problem
            "GDOP": [], # Store the Geometric Dilution of Precision of the linear least squares problem
            "residuals": [], # Store the residuals of the linear least squares problem
            "condition_number": [], # Store the condition number of the linear least squares problem
            "covariances": [], # Store the covariances of the linear least squares problem
            "verification_vector": [], # Store the verification vector of the linear least squares problem (the internal residual)
            "residual_vector": [], # Store the residual vector of the linear least squares problem
            "consecutive_distances_vector": [], # Store the consecutive distances between two estimates of the anchor position

            # Convergence counters - count how many times in a row the threshold is met by looking at deltas
            "FIM_convergence_counter": 0,
            "GDOP_convergence_counter": 0,
            "residuals_convergence_counter": 0,
            "condition_number_convergence_counter": 0,
            "covariances_convergence_counter": 0,
            "verification_vector_convergence_counter": 0,
            "consecutive_distances_vector_convergence_counter": 0,

            "GMM": []
        }

        # Dictionnary to store the measurements and estimations for each anchor using the template above
        self.anchor_measurements_dictionary = {}

        # Optimiser class instance to call to run the optimisation procedure
        self.trajectory_optimiser = TrajectoryOptimization(method="FIM", bounds=[(-1.5, 1.5), (-2.8, 2.8), (0.5, 2.5)], default_fim_noise_variance=0.4)

        # Trajectory to follow
        self.passed_waypoints = []
        self.remaining_waypoints = []

        self.current_optimal_waypoints = []
        self.current_link_waypoints = []

        self.status = "open_to_measurements" # "on_optimal_trajectory" 
        self.anchor_to_optimise_queue = {} # key is anchor id, data is estimated position

        # Tuning parameters for the pipeline
        self.params = {
            # Measurement gathering parameters
            'distance_to_anchor_ratio_threshold': 0.03, # tangent ratio between consecutive measurements -> smaller is more measurements
            'number_of_redundant_measurements': 1, # Number of measurements to take at the same place or consecutively, ignoring the condition above
            'distance_rejection_threshold': 15, # Minimum distance to the anchor to not reject the measurement
            
            # Least squares parameters
            'use_linear_bias': False, # Use linear bias in the linear least squares
            'use_constant_bias': False, # Use constant bias in the linear least squares
            'normalised': False, # Use the normalised formulation for the linear least squares

            'regularise': True, # Regularise the least squares problem to keep the biases small

            'rough_estimate_method': "linear_reweighted", # Method to use for the rough estimate, either simple_linear or linear_reweighted or em_algorithm
            'outlier_removing': "None", # "None", "counter", "immediate" - Method to use for outlier removal where counter removes after a certain number of consecutive outliers, immediate removes the measurement immediately
            'reweighting_iterations': 2, # Number of reweighting iterations for the reweighted least squares
            'use_trimmed_reweighted': True, # Use trimmed reweighted least squares for the rough estimate

            "weighting_function": "mad", # Method to use for the reweighting of the linear least squares problem, either huber, tukey, geman_mcclure, welsch, mad (best seem to be huber and geman_mcclure)
            'huber_delta': 0.01,
            'tukey_c': 4.685,
            'welsch_c': 2.0,

            'non_linear_optimisation_type': "EM", # Type of non linear optimisation to use, either IRLS or LM or KRR (does not work) or MM_Gaussian_Mixture

            # Stopping criterion parameters
            "stopping_criteria": ["nb_measurements"],

            # Absolute thresholds
            'FIM_thresh': 1e5,
            'GDOP_thresh': 5,
            'residuals_thresh': 100,
            'condition_number_thresh': 5e5,
            'covariance_thresh': 10,
            'verification_vector_thresh': 10,
            'number_of_measurements_thresh': 25,

            # Relative thresholds
            'FIM_ratio_thresh': 1,
            'GDOP_ratio_thresh': 0.2,
            'residuals_ratio_thresh': 1,
            'condition_number_ratio_thresh': 0.1,
            'covariance_ratio_thresh': 1,
            'convergence_postion_thresh': 1,
            'verification_vector_ratio_thresh': 0.1,

            # Convergence counters threshold - i.e how many times in a row the delta between two consecutive values for the criterion is below the threshold
            'convergence_counter_threshold': 3,

            # Outlier rejection parameters
            "z_score_threshold": 3,
            "outlier_count_threshold": 3, # Number of consecutive outliers to remove a measurement when using the counter method

            "trajectory_optimisation_method": "FIM", # Type of trajectory optimisation to use, either GDOP or FIM
            "link_method": "strict_return", 
        }
    

    def compute_GDOP(self, measurements, target_coords):
        """Compute the Geometric Dilution of Precision (GDOP) given a set of measurements corresponding to drone positions in space, and the target coordinates of the anchor to relate the measurements to
        
        Parameters:
        - measurements: list of lists, the measurements to use for the GDOP calculation where each line is of the form [x, y, z, distance]
        - target_coords: list of floats, the target coordinates of the anchor to relate the measurements to [x, y, z]
        
        Returns:
        - gdop: float, the computed GDOP, or infinity if the matrix is singular
        """
        
        if len(measurements) < 4:
            return float('inf') # Not enough points to calculate GDOP
    
        x,y,z = target_coords
        A = []
        for measurement in measurements:
            x_i, y_i, z_i, _ = measurement
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

    def compute_FIM(self, measurements, target_estimator, noise_variance):
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

        x0, y0, z0 = target_estimator[:3]
        measurements = np.array(measurements)
        measurements = measurements[:,:3] # Remove measurements with 0 distance
        noise_variance = 0.4 # self.fim_noise_variance # Default noise variance

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

    def compute_z_score(self, residuals):
        """Compute the z-score of the residuals from the linear least squares problem
        
        Parameters:
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - z_scores: numpy array, the computed z-scores
        """
        
        mean_residuals = np.mean(residuals)
        std_residuals = np.std(residuals)
        z_scores = (residuals - mean_residuals) / std_residuals

        return z_scores

    def compute_condition_number(self, measurements):
        """Compute the condition number of the linear least squares A matrix given a set of measurements
        
        Parameters:
        - measurements: list of lists, the measurements to use for the condition number calculation where each line is of the form [x, y, z, distance]
        
        Returns:
        - condition_number: float, the computed condition number
        """

        A, _ = self.setup_linear_least_square(measurements)
        # U, s, Vt = np.linalg.svd(A)
        # sigma_max = max(s)
        # sigma_min = min(s)
        # condition_number = sigma_max / sigma_min

        condition_number = np.linalg.cond(A)
        
        return condition_number

    def compute_mad(self, residuals):
        """Compute the Median Absolute Deviation (MAD) of the residuals from the linear least squares problem
        
        Parameters:
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - mad: float, the computed MAD
        """

        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))

        return mad



    def setup_linear_least_square(self, measurements):
        """Setup the linear least squares problem given a set of measurements, using the parameters set in the class for deciding on wether on not to include bias terms

        Parameters:
        - measurements: list of lists, the measurements to use for the linear least squares problem where each line is of the form [x, y, z, distance]

        Returns:
        - A: numpy array, the matrix A of the linear least squares problem
        - b: numpy array, the vector b of the linear least squares problem
        """

        use_linear_bias = self.params['use_linear_bias']
        use_constant_bias = self.params['use_constant_bias']

        # Normalised formulation (only works for both biases)
        if self.params['normalised']:
            A = []
            b = []

            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([2*x, 2*y, 2*z, measured_dist**2, -2*measured_dist, -norm_squared])
                b.append(1)

            return np.array(A), np.array(b)


        if use_constant_bias and use_linear_bias:
            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([2*x, 2*y, 2*z, measured_dist**2, -2*measured_dist, 1])
                b.append(norm_squared)

        elif use_constant_bias:
            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([-2*x, -2*y, -2*z, 2*measured_dist, 1])
                b.append(measured_dist**2 - norm_squared)

        elif use_linear_bias:
            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([2*x, 2*y, 2*z, measured_dist**2, 1])
                b.append(norm_squared)

        else:

            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([-2*x, -2*y, -2*z, 1])
                b.append(measured_dist**2 - norm_squared)


        return np.array(A), np.array(b)

    def retrieve_least_square_result(self, estimator):
        use_linear_bias = self.params['use_linear_bias']
        use_constant_bias = self.params['use_constant_bias']

        if use_constant_bias and use_linear_bias:

            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]
            gamma = estimator[3]
            beta = estimator[4]

            x = [x_a, y_a, z_a, 1/beta**2, gamma/beta**2, gamma**2/beta**2 - np.linalg.norm([x_a, y_a, z_a])**2]


        elif use_constant_bias:
                
            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]
            gamma = estimator[3]
            beta = 0

            x = [x_a, y_a, z_a, gamma, 1] # to be confirmed

        elif use_linear_bias:
                
            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]
            gamma = 0
            beta = estimator[3]

            x = [x_a, y_a, z_a, 1/beta**2, gamma/beta**2] # to be confirmed

        else:
                
            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]

            x = [x_a, y_a, z_a]

        return x

    def compute_verification_vector(self, estimator):

        use_linear_bias = self.params['use_linear_bias']
        use_constant_bias = self.params['use_constant_bias']
            
        if use_constant_bias and use_linear_bias:

            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]
            gamma = estimator[3]
            beta = estimator[4]

            x = [x_a, y_a, z_a, 1/beta**2, gamma/beta**2, gamma**2/beta**2 - np.linalg.norm([x_a, y_a, z_a])**2]


        elif use_constant_bias:
                
            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]
            gamma = estimator[3]
            p_a_gamma = estimator[4]
            
            norm = np.linalg.norm([x_a, y_a, z_a])

            return p_a_gamma - norm**2 + gamma**2

        elif use_linear_bias:
                    
            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]
            inv_beta_squared = estimator[3]
            neg_p_a = estimator[4]
            norm = np.linalg.norm([x_a, y_a, z_a])

            return norm**2 + neg_p_a

        else:
                        
            x_a = estimator[0]
            y_a = estimator[1]
            z_a = estimator[2]
            p_a = estimator[3]
            norm = np.linalg.norm([x_a, y_a, z_a])

            return p_a - norm**2
            
            

        

    def update_weights(self, residuals):
        """Update the weights based on the residuals of the previous estimation.
        
        Parameters:
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - weights: list, the updated weights
        """
        
        def compute_scale(residuals):

            mad = self.compute_mad(residuals)
            scale = mad / 0.6745

            return scale
        
        def psi_function(residuals, k = 1.345):
            return np.where(np.abs(residuals) <= k, residuals, k * np.sign(residuals))
        
        def psi_function_tukey(residuals, c = 4.685):
            return np.where(np.abs(residuals) <= c, residuals*(1 - (residuals/c)**2)**2, 0)
        
        if self.params['weighting_function'] == 'huber':
            delta = self.params['huber_delta']
            weights = np.where(np.abs(residuals) <= delta, 1, delta / np.abs(residuals)).tolist()

        elif self.params['weighting_function'] == 'tukey':
            c = self.params['tukey_c']
            weights = np.where(np.abs(residuals) <= c, (1 - (residuals/c)**2)**2, 0).tolist()

        elif self.params['weighting_function'] == 'geman_mcclure':
            weights = (1 / (1 + (residuals)**2)).tolist()

        elif self.params['weighting_function'] == 'welsch':
            c = self.params['welsch_c']
            weights = np.exp(-0.5 * (residuals/c)**2).tolist()
        
        elif self.params['weighting_function'] == 'mad':
            scale = compute_scale(residuals)
            weights = psi_function_tukey(residuals / scale)/(residuals/scale)
            weights = weights.tolist()
            # reject outliers
            # weights = np.where(np.abs(residuals/scale) > 3, 0, weights).tolist()
        else:
            weights = [1 / max(np.abs(res**2), 1e-8) for res in residuals]  # Default to inverse squared residuals with small constant
        
        return weights

    

    def estimate_anchor_position_linear_least_squares(self, measurements):
        """Estimate the position of an anchor given distance measurements fron different drone positions using linear least squares optimization depending on the parameters set in the class for bias usage
        
        Parameters:
        - measurements: list of lists, the measurements to use for the linear least squares problem where each line is of the form [x, y, z, distance]
        
        Returns:
        - estimator: numpy array, the estimated position of the anchor and the bias terms if used
        - covariance_matrix: numpy array, the covariance matrix of the linear least squares estimation
        - residuals: numpy array, the residuals of the linear least squares estimation
        - x: numpy array, the estimated parameters of the linear least squares estimation, as it is setup
        """

        use_constant_bias = self.params['use_constant_bias']
        use_linear_bias = self.params['use_linear_bias']
    
        def compute_covariance_matrix_linear_least_squares(A, b, x):
            """Compute the covariance matrix of the parameters for linear least squares
            
            Parameters:
            - A: numpy array, the matrix A of the linear least squares problem
            - b: numpy array, the vector b of the linear least squares problem
            - x: numpy array, the estimated parameters of the linear least squares problem
            
            Returns:
            - covariance_matrix: numpy array, the computed covariance matrix
            """
            
            residual_sum_of_squares = np.sum((b - A @ x)**2)
            dof = A.shape[0] - A.shape[1]
            sigma_hat_squared = residual_sum_of_squares / dof
            covariance_matrix = sigma_hat_squared * np.linalg.pinv(A.T @ A)

            return covariance_matrix
        
        def compute_residuals(A, b, x):
            """Compute the residuals of the linear least squares problem given the matrix A, the vector b and the estimated parameters x

            Parameters:
            - A: numpy array, the matrix A of the linear least squares problem
            - b: numpy array, the vector b of the linear least squares problem
            - x: numpy array, the estimated parameters of the linear least squares problem

            Returns:
            - residuals: numpy array, the computed residuals
            """

            return np.array((b - A @ x))

        A, b = self.setup_linear_least_square(measurements)

        # Normalised formulation, only works for both biases
        if self.params['normalised']:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            squared_norm_of_anchor = x[5]
            position = x[:3] / squared_norm_of_anchor

            beta_squared = 1/(x[3]*squared_norm_of_anchor)

            bias = x[4]/x[3]
            linear_bias = np.sqrt(beta_squared)

            return np.concatenate((position, [bias, linear_bias])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x

        if use_constant_bias and use_linear_bias:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            squared_linear_bias = x[3] if x[3] > 0 else 1
            linear_bias = np.sqrt(1/squared_linear_bias)

            bias = x[4]/squared_linear_bias
            position = x[:3]

            return np.concatenate((position, [bias, linear_bias])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x
        
        elif use_constant_bias:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            position = x[:3]
            bias = x[3]

            return np.concatenate((position, [bias, 1])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x
        
        elif use_linear_bias:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            squared_linear_bias = x[3] if x[3] > 0 else 1
            linear_bias = np.sqrt(1/squared_linear_bias)
            position = x[:3]

            return np.concatenate((position, [0, linear_bias])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x
        
        else:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            position = x[:3]

            return np.concatenate((position, [0, 1])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x

    def estimate_anchor_position_weighted_linear_least_squares(self, measurements, anchor_id, A = None, b = None, weights = None):


        use_constant_bias = self.params['use_constant_bias']
        use_linear_bias = self.params['use_linear_bias']
        regularise = self.params['regularise']

        
    
        def compute_covariance_matrix__weighted_linear_least_squares(A, b, x, W):
            """Compute the covariance matrix of the parameters for linear least squares
            
            Parameters:
            - A: numpy array, the matrix A of the linear least squares problem
            - b: numpy array, the vector b of the linear least squares problem
            - x: numpy array, the estimated parameters of the linear least squares problem
            
            Returns:
            - covariance_matrix: numpy array, the computed covariance matrix
            """

            residual_sum_of_squares = np.sum(((b - A @ x)**2) * np.diag(W))
            dof = A.shape[0] - A.shape[1]
            sigma_hat_squared = residual_sum_of_squares / dof
            covariance_matrix = sigma_hat_squared * np.linalg.pinv(A.T @ W @ A)

            return covariance_matrix
        
        def compute_residuals(A, b, x):
            """Compute the residuals of the linear least squares problem given the matrix A, the vector b and the estimated parameters x

            Parameters:
            - A: numpy array, the matrix A of the linear least squares problem
            - b: numpy array, the vector b of the linear least squares problem
            - x: numpy array, the estimated parameters of the linear least squares problem

            Returns:
            - residuals: numpy array, the computed residuals
            """

            return np.array((b - A @ x))
        
        def compute_residuals_true(measurements, estimator):
            residuals = []
            for measurement in measurements:
                x,y,z, range_measurement = measurement
                distance = np.linalg.norm([x, y, z] - estimator[:3])
                residuals.append(range_measurement - estimator[4]* distance - estimator[3])
            return np.array(residuals)
        

        if A is None or b is None:
            A, b = self.setup_linear_least_square(measurements)
        
        if weights is None:
            if len(self.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"][-1]) == 0:
                weights = np.ones(len(measurements))
            else:
                weights = self.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"][-1]
                # A_new, b_new = self.setup_linear_least_square([measurements[-1]])
                # estimator_previous = self.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"]
                # x_previous = self.retrieve_least_square_result(estimator_previous)
                # new_measurement_residual = b_new - A_new @ x_previous
                new_measurement_weight = 1 #self.update_weights_huber(new_measurement_residual)[0] # or 1
                
                weights.append(new_measurement_weight)
            weights = np.ones(len(measurements))

        # W = np.sqrt(np.diag(weights)) # Create diagonal matrix of weights

        # Normalised formulation, only works for both biases
        if self.params['normalised']:

            reg_matrix = np.eye(A.shape[1])
            lambda_vector = np.zeros(A.shape[1])
            
            if regularise:
                lambda_vector[2] = 0
                lambda_vector[3] = 100
                lambda_vector[4] = 100
                lambda_vector[5] = 1
                reg_matrix = reg_matrix * lambda_vector
            else:
                reg_matrix = np.zeros(A.shape[1])



            W = np.diag(weights)
            x = np.linalg.pinv(A.T @ W @ A + reg_matrix) @ (A.T @ W @ b)

            squared_norm_of_anchor = 1/x[5]
            position = x[:3] * squared_norm_of_anchor

            beta_squared = 1/(x[3]*squared_norm_of_anchor)
            
            bias = x[4]/x[3]
            linear_bias = (np.sqrt(beta_squared) if beta_squared > 0 else 1)

            estimator = np.concatenate((position, [bias, linear_bias]))
            return estimator, compute_covariance_matrix__weighted_linear_least_squares(A, b, x, W), compute_residuals_true(measurements, estimator), x # compute_residuals(A, b, x), x
        
        if use_constant_bias and use_linear_bias:
            
            reg_matrix = np.eye(A.shape[1])
            lambda_vector = np.zeros(A.shape[1])
            
            if regularise:
                # lambda_vector[2] = 100
                lambda_vector[3] = 10
                lambda_vector[4] = 10
                reg_matrix = reg_matrix * lambda_vector
            else:
                reg_matrix = np.zeros(A.shape[1])

    
            W = np.diag(weights)
            x = np.linalg.pinv(A.T @ W @ A + reg_matrix) @ (A.T @ W @ b)
            squared_linear_bias = x[3] if x[3] > 0 else 1
            linear_bias = np.sqrt(1/squared_linear_bias)

            bias = x[4]/squared_linear_bias
            position = x[:3]
            estimator = np.concatenate((position, [bias, linear_bias]))
            return estimator, compute_covariance_matrix__weighted_linear_least_squares(A, b, x, W), compute_residuals(A,b,x), x
        
        elif use_constant_bias:

            reg_matrix = np.eye(A.shape[1])
            lambda_vector = np.zeros(A.shape[1])
            
            if regularise:
                # lambda_vector[2] = 100
                lambda_vector[3] = 100
                reg_matrix = reg_matrix * lambda_vector
            else:
                reg_matrix = np.zeros(A.shape[1])

            # Weighted least squares: x = (A^T W A)^{-1} A^T W b
            W = np.diag(weights)
            x = np.linalg.pinv(A.T @ W @ A + reg_matrix) @ (A.T @ W @ b)


            position = x[:3]
            bias = x[3]

            return np.concatenate((position, [bias, 1])), compute_covariance_matrix__weighted_linear_least_squares(A, b, x, W), compute_residuals(A, b, x), x
        
        elif use_linear_bias:

            reg_matrix = np.eye(A.shape[1])
            lambda_vector = np.zeros(A.shape[1])
            
            if regularise:
                # lambda_vector[2] = 100
                lambda_vector[3] = 100
                reg_matrix = reg_matrix * lambda_vector
            else:
                reg_matrix = np.zeros(A.shape[1])

            # Weighted least squares: x = (A^T W A)^{-1} A^T W b
            W = np.diag(weights)
            x = np.linalg.pinv(A.T @ W @ A + reg_matrix) @ (A.T @ W @ b)

            squared_linear_bias = x[3] if x[3] > 0 else 1
            linear_bias = np.sqrt(1/squared_linear_bias)
            position = x[:3]

            return np.concatenate((position, [0, linear_bias])), compute_covariance_matrix__weighted_linear_least_squares(A, b, x, W), compute_residuals(A, b, x), x
        
        else:


            # Weighted least squares: x = (A^T W A)^{-1} A^T W b
            W = np.diag(weights)
            x = np.linalg.pinv(A.T @ W @ A) @ (A.T @ W @ b)

            position = x[:3]

            return np.concatenate((position, [0, 1])), compute_covariance_matrix__weighted_linear_least_squares(A, b, x, W), compute_residuals(A, b, x), x

    def estimate_anchor_position_trimmed_linear_least_squares(self, measurements, anchor_id, trim_fraction=0.1):

        """
        Combine Trimmed Least Squares with Weighted Least Squares.

        Parameters:
        - A: numpy array, the design matrix (m x n).
        - b: numpy array, the observation vector (m x 1).
        - weights: numpy array, the weights for each measurement (m x 1).
        - trim_fraction: float, the fraction of measurements to trim (e.g., 0.1 for 10% trimming).

        Returns:
        - x_final: numpy array, the final estimated parameters after trimming and WLS.
        """

        A, b = self.setup_linear_least_square(measurements)
        weights = self.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"][-1]
        weights = np.array(weights)

        # Step 3: Trim the top trim_fraction of residuals
        threshold = np.percentile(weights, 100 * (1 - trim_fraction))
        mask = weights <= threshold
        
        # Apply mask to A, b, and weights
        A_trimmed, b_trimmed, weights_trimmed = A[mask], b[mask], weights[mask]
        
        # Step 4: Perform final WLS on the trimmed dataset
        x_final, covariances, residuals_final, x = self.estimate_anchor_position_weighted_linear_least_squares(measurements, anchor_id, A_trimmed, b_trimmed, weights_trimmed)
        
        return x_final, covariances, residuals_final, x
       
    def estimate_anchor_position_non_linear_least_squares(self, measurements, initial_guess=[0, 0, 0, 0, 0], max_iterations=20, tol=1e-6, alpha=1.0, sigma=1.0):
        """Estimate the position of an anchor given distance measurements fron different drone positions using non lienar ptimization
        
        Parameters:
        - measurements: list of lists, the measurements to use for the non linear least squares problem where each line is of the form [x, y, z, distance]
        - initial_guess: list of floats, the initial guess for the non linear least squares optimization, i.e [x, y, z, bias, linear_bias]
        - max_iterations: int, the maximum number of iterations for the optimization if using IRLS
        - tol: float, the tolerance for the optimization if using IRLS
        - alpha: float, the regularization parameter for the KRR model
        - sigma: float, the kernel width for the KRR model

        Returns:
        - estimator: numpy array, the estimated position of the anchor and the bias terms if used
        - covariance_matrix: numpy array, the covariance matrix of the non linear least squares estimation
        """
        
        measurements = np.array(measurements)
        optimisation_type = self.params["non_linear_optimisation_type"]

        if optimisation_type == "LM":
            # Define the loss function and its Jacobian
            def loss_function(x0, measurements):
                residuals = []
                J = []  # Jacobian matrix
                guess = x0[0:3]
                bias = x0[3]
                linear_bias = x0[4]
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[0:4]
                    distance_vector = np.array([guess[0] - x, guess[1] - y, guess[2] - z])
                    distance = np.linalg.norm(distance_vector)
                    estimated_dist = distance * (linear_bias) + bias
                    residual = estimated_dist - measured_dist
                    residuals.append(residual)

                    # Jacobian calculation
                    if distance != 0:  # to avoid division by zero
                        J_row = np.zeros(5)
                        J_row[0:3] = (linear_bias) * (distance_vector / distance)
                        J_row[3] = 1  # bias term
                        J_row[4] = distance  # linear_bias term
                        J.append(J_row)

                return residuals, J

            # Perform least squares optimization and calculate Jacobian at solution
            result = least_squares(
            lambda x0, measurements: loss_function(x0, measurements)[0],
            initial_guess,
            args=(measurements,),
            jac=lambda x0, measurements: np.array(loss_function(x0, measurements)[1]),
            method='lm')
            
            # Calculate covariance matrix from the Jacobian at the last evaluated point
            J = np.array(loss_function(result.x, measurements)[1])

            # Residual sum of squares
            residual_sum_of_squares = np.sum(result.fun**2)

            # Degrees of freedom: number of data points minus number of parameters
            dof = len(measurements) - len(result.x)

            # Estimate of the error variance
            sigma_hat_squared = residual_sum_of_squares / dof

            # Covariance matrix of the parameters
            cov_matrix = sigma_hat_squared * np.linalg.inv(J.T @ J)


            return result.x, cov_matrix
        
        elif optimisation_type == "IRLS":

            def loss_function(params, measurements):
                x_a, y_a, z_a, gamma, beta = params
                anchor_pos = np.array([x_a, y_a, z_a])
                residuals = []
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[:4]
                    distance_vector = anchor_pos - np.array([x, y, z])
                    distance = np.linalg.norm(distance_vector)
                    estimated_dist = beta * distance + gamma
                    residual = measured_dist - estimated_dist
                    residuals.append(residual)
                return np.array(residuals)
            
            def jacobian(params, measurements):
                x_a, y_a, z_a, gamma, beta = params
                anchor_pos = np.array([x_a, y_a, z_a])
                J = []
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[:4]
                    distance_vector = anchor_pos - np.array([x, y, z])
                    distance = np.linalg.norm(distance_vector)
                    if distance == 0:
                        continue
                    J_row = np.zeros(5)
                    J_row[0:3] = -beta * (distance_vector / distance)
                    J_row[3] = -1
                    J_row[4] = -distance
                    J.append(J_row)
                return np.array(J)
            
            def compute_mad(residuals):
                median = np.median(residuals)
                mad = np.median(np.abs(residuals - median))
                return mad

            def compute_scale(residuals):
                mad = np.median(np.abs(residuals - np.median(residuals)))
                scale = 1.4826 * mad
                return scale

            def huber_weights_function(residuals, scale, c=1.345):
                """
                Compute Huber weights for the residuals.
                
                Parameters:
                - residuals: array-like, the residuals of the model.
                - scale: float, the estimated scale (usually MAD-based).
                - c: float, the threshold parameter where the function switches from quadratic to linear.
                
                Returns:
                - weights: array-like, the computed weights.
                """
                scaled_residuals = residuals / scale
                weights = np.where(np.abs(scaled_residuals) <= c, 1, c / np.abs(scaled_residuals))
                return weights
            
            def tukey_weights_function(residuals, scale, c=4.685):
                """
                Compute Tukey's biweight (bisquare) weights for the residuals.
                
                Parameters:
                - residuals: array-like, the residuals of the model.
                - scale: float, the estimated scale (usually MAD-based).
                - c: float, the tuning constant (often 4.685 for 95% efficiency).
                
                Returns:
                - weights: array-like, the computed weights.
                """
                scaled_residuals = residuals / scale
                abs_scaled_residuals = np.abs(scaled_residuals)
                weights = np.zeros_like(residuals)
                mask = abs_scaled_residuals <= c
                weights[mask] = (1 - (scaled_residuals[mask] / c) ** 2) ** 2
                return weights

            def cauchy_weights_function(residuals, scale, c=2.3849):
                """
                Compute Cauchy weights for the residuals.
                
                Parameters:
                - residuals: array-like, the residuals of the model.
                - scale: float, the estimated scale (usually MAD-based).
                - c: float, the tuning constant (adjust according to the specific problem).
                
                Returns:
                - weights: array-like, the computed weights.
                """
                scaled_residuals = residuals / scale
                weights = 1 / (1 + (scaled_residuals / c) ** 2)
                return weights

            params = np.array(initial_guess)
            
            lower_bounds = [-np.inf, -np.inf, 0, 0, 1]  # x_a, y_a, z_a >= 0; gamma, beta unbouded
            upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]  # No upper limits

            for i in range(max_iterations):

                residuals = loss_function(params, measurements)

                # Compute the scale (MAD-based)
                scale = compute_scale(residuals)
                
                # Use Tukey's biweight function for weights
                weights = tukey_weights_function(residuals, scale, c=4.685)

                sqrt_weights = np.sqrt(weights)
                
                weighted_residuals = sqrt_weights * residuals
                weighted_J = sqrt_weights[:, np.newaxis] * jacobian(params, measurements)
                
                # Perform the weighted least squares optimization
                result = least_squares(lambda p: sqrt_weights * loss_function(p, measurements), params, jac=lambda p: sqrt_weights[:, np.newaxis] * jacobian(p, measurements),method='lm') #'trf', bounds=(lower_bounds, upper_bounds))
                
                new_params = result.x
                if np.linalg.norm(new_params - params) < tol:
                    print(f"Converged after {i} iterations")
                    break
                params = new_params
            
            # Compute the final covariance matrix
            J_final = jacobian(params, measurements)
            residual_sum_of_squares = np.sum(loss_function(params, measurements)**2)
            dof = len(measurements) - len(params)
            sigma_hat_squared = residual_sum_of_squares / dof
            cov_matrix = sigma_hat_squared * np.linalg.pinv(J_final.T @ J_final)
            

            return params, cov_matrix

        elif optimisation_type == "MM":
            
            def loss_function(params, measurements):
                x_a, y_a, z_a, gamma, beta = params
                anchor_pos = np.array([x_a, y_a, z_a])
                residuals = []
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[:4]
                    distance_vector = anchor_pos - np.array([x, y, z])
                    distance = np.linalg.norm(distance_vector)
                    estimated_dist = beta * distance + gamma
                    residual = measured_dist - estimated_dist
                    residuals.append(residual)
                return np.array(residuals)
            
            def jacobian(params, measurements):
                x_a, y_a, z_a, gamma, beta = params
                anchor_pos = np.array([x_a, y_a, z_a])
                J = []
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[:4]
                    distance_vector = anchor_pos - np.array([x, y, z])
                    distance = np.linalg.norm(distance_vector)
                    if distance == 0:
                        continue
                    J_row = np.zeros(5)
                    J_row[0:3] = -beta * (distance_vector / distance)
                    J_row[3] = -1
                    J_row[4] = -distance
                    J.append(J_row)
                return np.array(J)
            
            def tukey_loss(residuals, scale, c=4.685):
                scaled_residuals = residuals / scale
                abs_scaled_residuals = np.abs(scaled_residuals)
                loss = np.zeros_like(scaled_residuals)

                mask = abs_scaled_residuals <= c
                loss[mask] = (scaled_residuals[mask]**2 / 2) * (1 - (scaled_residuals[mask] / c)**2 + (scaled_residuals[mask] / c)**4 / 3)
                loss[~mask] = c**2 / 6
                return np.sum(loss)

            def compute_mad(residuals):
                median = np.median(residuals)
                mad = np.median(np.abs(residuals - median))
                return mad

            def loss_function_s_estimator(params, measurements, c=4.685):
                x_a, y_a, z_a, gamma, beta = params[:5]
                scale = params[5]
                
                anchor_pos = np.array([x_a, y_a, z_a])
                residuals = []
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[:4]
                    distance_vector = anchor_pos - np.array([x, y, z])
                    distance = np.linalg.norm(distance_vector)
                    estimated_dist = beta * distance + gamma
                    residual = measured_dist - estimated_dist
                    residuals.append(residual)
                
                residuals = np.array(residuals)
                
                return tukey_loss(residuals, scale, c)

            def s_estimator_optimization(initial_guess, measurements):
                # Adding an initial guess for scale, set to some initial value like MAD
                mad_initial = compute_mad(loss_function(initial_guess, measurements))
                initial_params = np.append(initial_guess, mad_initial)
                
                result = minimize(loss_function_s_estimator, initial_params, args=(measurements,),
                                method='L-BFGS-B',
                                bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), 
                                        (0.1, np.inf), (1e-6, np.inf)])  # bounds ensure scale is positive
                params = result.x[:-1]  # Extract the optimized parameters
                scale = result.x[-1]    # Extract the optimized scale
                
                return params, scale

            def final_refined_estimation(params, measurements, initial_scale, c_final=1.345):
                """
                Refine estimates using a more efficient M-estimator with a lower tuning constant.
                """
                def huber_weights_function(residuals, scale, c):
                    scaled_residuals = residuals / scale
                    abs_scaled_residuals = np.abs(scaled_residuals)
                    weights = np.where(abs_scaled_residuals <= c, 1, c / abs_scaled_residuals)
                    return weights

                for i in range(max_iterations):
                    residuals = loss_function(params, measurements)

                    # Use the scale from the initial robust estimator
                    weights = huber_weights_function(residuals, initial_scale, c_final)
                    sqrt_weights = np.sqrt(weights)

                    result = least_squares(lambda p: sqrt_weights * loss_function(p, measurements),
                                        params, 
                                        jac=lambda p: sqrt_weights[:, np.newaxis] * jacobian(p, measurements),
                                        method='lm')
                    
                    new_params = result.x
                    if np.linalg.norm(new_params - params) < tol:
                        break
                    params = new_params

                return params
            
            # Initial robust estimation using S-estimator
            initial_params, initial_scale = s_estimator_optimization(initial_guess, measurements)

            # Final efficient M-estimator refinement
            params = final_refined_estimation(initial_params, measurements, initial_scale)
            
            # Compute covariance matrix as before
            J_final = jacobian(params, measurements)
            residual_sum_of_squares = np.sum(loss_function(params, measurements)**2)
            dof = len(measurements) - len(params)
            sigma_hat_squared = residual_sum_of_squares / dof
            cov_matrix = sigma_hat_squared * np.linalg.pinv(J_final.T @ J_final)

            return params, cov_matrix
        
        elif optimisation_type == "EM":

            
            def gaussian_likelihood(residuals, sigma, mu):
                return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((residuals - mu) / sigma) ** 2)

            # GMM likelihood function
            def gmm_likelihood(params, measurements):
                # Extract parameters from the combined vector
                object_pos = params[:3]
                gamma = params[3]
                beta = params[4]
                pi_los = params[5]
                sigma_los = params[6]
                sigma_nlos = params[7]
                mu_los = params[8]
                mu_nlos = params[9]

                n = len(measurements)
                log_likelihood = 0

                pi_los = np.clip(pi_los, 0.01, 0.99)  # Keep between 0.01 and 0.99
                pi_nlos = 1.0 - pi_los

                

                for i in range(n):
                    d_i = np.linalg.norm(measurements[i][:3] - object_pos)
                    
                    # LOS Gaussian component
                    los_prob = pi_los * gaussian_likelihood(measurements[i][3]-d_i*beta-gamma, sigma_los, mu_los)
                    
                    # NLOS Gaussian component (with higher mean)
                    nlos_prob = pi_nlos * gaussian_likelihood(measurements[i][3]-d_i*beta-gamma, sigma_nlos, mu_nlos)
                    
                    # Mixture model probability
                    total_prob = los_prob + nlos_prob
                    total_prob = np.maximum(total_prob, 1e-6)  # Avoid division by zero
                    log_likelihood += np.log(total_prob)
                    
                return -log_likelihood  # We minimize the negative log-likelihood
            
            def gmm_likelihood_distance_dependant(params, measurements):
                # Extract parameters from the combined vector
                object_pos = params[:3]
                beta = params[3]               # Beta remains for scaling distances
                pi_los = params[4]
                sigma_0_los = params[5]
                alpha_los = params[6]
                sigma_0_nlos = params[7]
                alpha_nlos = params[8]
                mu_los = params[9]
                mu_nlos = params[10]

                n = len(measurements)
                log_likelihood = 0

                # Ensure pi_los is within [0.01, 0.99]
                pi_los = np.clip(pi_los, 0.01, 0.99)
                pi_nlos = 1.0 - pi_los

                # Compute residuals without gamma
                sensor_positions = measurements[:, :3]
                ranges = measurements[:, 3]
                distances = np.linalg.norm(sensor_positions - object_pos, axis=1)
                
                residuals = ranges - distances * beta  # Removed gamma

                # Distance-dependent noise for LOS and NLOS
                sigma_los = sigma_0_los + alpha_los * distances
                sigma_nlos = sigma_0_nlos + alpha_nlos * distances

                # LOS and NLOS probabilities
                los_prob = pi_los * gaussian_likelihood(residuals, sigma_los, mu_los)
                nlos_prob = pi_nlos * gaussian_likelihood(residuals, sigma_nlos, mu_nlos)

                # Mixture model probability
                total_prob = los_prob + nlos_prob
                total_prob = np.maximum(total_prob, 1e-6)  # Avoid division by zero
                # Sum of log-likelihoods
                log_likelihood = np.sum(np.log(total_prob))

                return -log_likelihood  # We minimize the negative log-likelihood

            # Initial guesses for the GMM parameters
            initial_object_pos = initial_guess[:3]
            initial_guess_gamma = initial_guess[3]
            initial_guess_beta = initial_guess[4]

            def compute_residuals(params, measurements):
                # Extract parameters from the combined vector
                object_pos = params[:3]
                gamma = params[3]
                beta = params[4]
                residuals = []
                for measurement in measurements:
                    d_i = np.linalg.norm(measurement[:3] - object_pos)
                    residual = measurement[3] - d_i * beta - gamma
                    residuals.append(residual)
                return np.array(residuals)
            
            residuals = compute_residuals(initial_guess, measurements)

            outliers = self.outlier_finder(residuals)
            outlier_mask = np.zeros(len(residuals), dtype=bool)
            outlier_mask[outliers] = True
            
            # los_measurements = [measurements[i] for i in range(len(measurements)) if i not in outliers]
            # nlos_measurements = [measurements[i] for i in range(len(measurements)) if i in outliers]

            
            if len(outliers) > 0:
                mu_los = np.mean(residuals[~outlier_mask])
                mu_nlos = np.mean(residuals[outlier_mask])
                sigma_los = np.std(residuals[~outlier_mask])
                sigma_nlos = np.std(residuals[outlier_mask])
                pi_los = 1 - len(outliers) / len(measurements)
            else:
                mu_los = np.mean(residuals)
                mu_nlos = np.mean(residuals)
                sigma_los = np.std(residuals)
                sigma_nlos = np.std(residuals)
                pi_los = 1.0

            # Separate in LOS and NLOS measurements
            # Use this to determine the pi_los, and then the sigma_los and sigma_nlos and mu_los and mu_nlos

            # sigma_los = 0.5  # Initial guess for LOS noise standard deviation
            # sigma_nlos = 1.0  # Initial guess for NLOS noise standard deviation
            # mu_los = 0.0
            # mu_nlos = 2.0  # Initial guess for NLOS mean


            initial_params = np.hstack([initial_object_pos, initial_guess_gamma, initial_guess_beta, pi_los, sigma_los, sigma_nlos, mu_los, mu_nlos])
            
            # Optimization
            result = minimize(
                gmm_likelihood,
                initial_params,
                args=(measurements),
                method='L-BFGS-B',
                bounds=[(None, None), (None, None), (0, None), (-0.1, 0.1), (0.9, 1.1),(0.65, 1.0), (0.01, None), (0.02, None), (0, 0),(-3, 3)]
            )

            # Estimated object position
            estimated_position = result.x[:3]

            
            print("Result of the gaussian mixture model optimization:")
            print("gamma:", result.x[3])
            print("beta:", result.x[4])
            print("pi_los:", result.x[5])
            print("sigma_los:", result.x[6])
            print("sigma_nlos:", result.x[7])
            print("mu_los:", result.x[8])
            print("mu_nlos:", result.x[9])
            
            initial_params = np.hstack([initial_object_pos, initial_guess_beta, pi_los, sigma_los, 1, sigma_nlos, 1, mu_los, mu_nlos])
            result = minimize(
                gmm_likelihood_distance_dependant,
                initial_params,
                args=(measurements),
                method='L-BFGS-B',
                bounds=[(None, None), (None, None), (0, None), (0, None), (0.5, 1.0), (0.01, None), (0.0, None), (0.01, None), (0.0, None), (0, None), (None, None)]
            )


            # estimated_position = result.x[:3]
            return np.concatenate((estimated_position, [0,1])), np.zeros((3,3))
        




        elif optimisation_type == "EM_new":

            def gaussian_pdf(x, mu, sigma):
                return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            
            def distance_dependant_noise_model(distance, sigma_0, alpha):
                return sigma_0 + distance*alpha
            
            def log_likelihood(params, measurements):
                """
                Compute the log-likelihood of the data given the parameters for LoS and NLoS components.
                
                Parameters:
                - params: list or numpy array containing the parameters [x0, y0, z0, mu_los, mu_nlos, sigma_los, sigma_nlos, alpha_los, alpha_nlos, pi_los, pi_nlos]
                - measurements: numpy array of shape (n, 4), where each row is [x, y, z, range]
                
                Returns:
                - log_likelihood: scalar, the computed log-likelihood value.
                """
                x0, y0, z0, mu_los, mu_nlos, sigma_los, sigma_nlos, alpha_los, alpha_nlos, logit_pi_los, logit_pi_nlos = params
                
                # Extract measurement data
                measurement_positions = measurements[:, :3]  # [x, y, z]
                ranges = measurements[:, 3]  # [range]
                
                pi_los = 1 / (1 + np.exp(-logit_pi_los))
                pi_nlos = 1 / (1 + np.exp(-logit_pi_nlos))

                # Ensure they sum to 1
                pi_sum = pi_los + pi_nlos
                pi_los /= pi_sum
                pi_nlos /= pi_sum

                # Compute log-likelihood
                log_likelihood = 0.0
                n = len(measurements)
                
                for i in range(n):
                    x_i = measurement_positions[i]
                    r_i = ranges[i]
                    
                    # Compute the Euclidean distance from the object's position
                    distance = np.linalg.norm([x_i[0] - x0, x_i[1] - y0, x_i[2] - z0])
                    error = distance - r_i
                    
                    # Compute noise standard deviations
                    sigma_los_actual = distance_dependant_noise_model(distance, sigma_los, alpha_los)
                    sigma_nlos_actual = distance_dependant_noise_model(distance, sigma_nlos, alpha_nlos)
                    
                    # Compute the likelihoods for LoS and NLoS
                    likelihood_los = gaussian_pdf(error, mu_los, sigma_los_actual)
                    likelihood_nlos = gaussian_pdf(error, mu_nlos, sigma_nlos_actual)
                    
                    # Combine the likelihoods using the mixing coefficients
                    weighted_likelihood = pi_los * likelihood_los + pi_nlos * likelihood_nlos
                    
                    # Avoid log(0) by adding a small constant
                    log_likelihood += np.log(weighted_likelihood + 1e-10)
                
                return -log_likelihood  # Return negative log-likelihood for minimization
            

            def optimize_gmm_parameters(measurements, initial_params):
                """
                Optimize the parameters of the GMM and object position.
                
                Parameters:
                - measurements: numpy array of shape (n, 4), where each row is [x, y, z, range]
                - initial_params: list or numpy array containing initial guesses for [x0, y0, z0, mu_los, mu_nlos, sigma_los, sigma_nlos, alpha_los, alpha_nlos, pi_los, pi_nlos]
                
                Returns:
                - result: scipy.optimize.OptimizeResult, result of the optimization.
                """
                bounds = [
                    (None, None),  # x0
                    (None, None),  # y0
                    (None, None),  # z0
                    (None, None),  # mu_los
                    (None, None),  # mu_nlos
                    (0.01, None),  # sigma_los
                    (0.01, None),  # sigma_nlos
                    (0.0, None),  # alpha_los
                    (0.0, None),  # alpha_nlos
                    (0, 1),  # pi_los
                    (0, 1)   # pi_nlos
                ]
                result = minimize(log_likelihood, initial_params, args=(measurements,), method='L-BFGS-B', bounds=bounds)
                
                return result.x

            # Example usage
            initial_params = [initial_guess[0], initial_guess[1], initial_guess[2], 0, 0, 0.01, 0.01, 1, 1, 0.9, 0.1]  # Example initial parameters

            optimized_params = optimize_gmm_parameters(measurements, initial_params)



            return np.concatenate((optimized_params[:3], [0, 1])), np.zeros((3,3))

                    




        elif optimisation_type == "EM_iterative":
            
            def compute_distances(measurements, x_obj, y_obj, z_obj):
                """
                Computes the distance between the object and the sensor positions.
                """
                distances = np.sqrt((measurements[:, 0] - x_obj)**2 + 
                                    (measurements[:, 1] - y_obj)**2 + 
                                    (measurements[:, 2] - z_obj)**2)
                return distances

            # Function to fit a GMM to the residuals
            def fit_gmm(residuals, n_components=3):
                """
                Fit a Gaussian Mixture Model (GMM) to the residuals.
                """
                pass
                # gmm = GaussianMixture(n_components=n_components, covariance_type='full')
                # gmm.fit(residuals.reshape(-1, 1))  # GMM expects a 2D array, reshape the residuals
                # return gmm

            # Function to update the object's position using weighted least squares
            def update_position(measurements, gmm, responsibilities):
                """
                Update the object position using a weighted least squares approach.
                """
                def objective_function(position):
                    x_obj, y_obj, z_obj = position
                    predicted_ranges = compute_distances(measurements, x_obj, y_obj, z_obj)
                    residuals = measurements[:, 3] - predicted_ranges
                    weights = responsibilities[:, 0]  # LoS responsibilities (first component)
                    return np.sum(weights * residuals**2)  # Weighted least squares

                # Use scipy minimize to find the optimal position
                initial_position = np.array([x0, y0, z0])
                result = minimize(objective_function, initial_position, method='L-BFGS-B')
                
                return result.x  # Return the updated object position

            # Main function implementing the iterative EM algorithm
            def em_algorithm(measurements, x0, y0, z0, max_iterations=100, tol=1e-5):
                """
                The two-step iterative EM algorithm for GMM + object position estimation.
                """
                # Initialize object position
                x_obj, y_obj, z_obj = x0, y0, z0
                n_measurements = measurements.shape[0]
                
                for iteration in range(max_iterations):
                    # Step 1: Compute residuals (difference between measured and predicted range)
                    predicted_ranges = compute_distances(measurements, x_obj, y_obj, z_obj)
                    residuals = measurements[:, 3] - predicted_ranges
                    
                    # Step 2: Fit GMM to residuals
                    gmm = fit_gmm(residuals, n_components=3)
                    
                    # Step 3: Compute the responsibilities (probabilities for each component)
                    responsibilities = gmm.predict_proba(residuals.reshape(-1, 1))
                    
                    # Step 4: Update the object position using the responsibilities (weighted least squares)
                    new_position = update_position(measurements, gmm, responsibilities)
                    
                    # Convergence check
                    position_change = np.linalg.norm(new_position - np.array([x_obj, y_obj, z_obj]))
                    if position_change < tol:
                        print(f"Converged after {iteration + 1} iterations")
                        break
                    
                    # Update the object position for the next iteration
                    x_obj, y_obj, z_obj = new_position
                
                # Final object position after EM algorithm
                return x_obj, y_obj, z_obj, gmm

            x0, y0, z0 = initial_guess[:3]

            final_x, final_y, final_z, gmm = em_algorithm(measurements, x0, y0, z0)

            return np.array([final_x, final_y, final_z, 0, 1]), np.zeros((3,3))

        elif optimisation_type == "EM_combined":

            def univariate_gaussian_pdf(x, mu, sigma):
                """
                Compute the probability density function of a univariate Gaussian distribution.
                
                Parameters:
                x : array-like
                    Data points (residuals).
                mu : float
                    Mean of the Gaussian distribution.
                sigma : float
                    Standard deviation of the Gaussian distribution.
                
                Returns:
                pdf : array-like
                    Probability density function values for each data point in x.
                """
                variance = sigma ** 2
                pdf = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mu) ** 2 / variance))
                return pdf

            # For multivariate Gaussian with scalar covariances, this can be simplified to univariate
            def multivariate_gaussian_pdf(x, mu, sigma):
                """
                Compute the probability density function of a multivariate Gaussian distribution.
                Here, it's implemented for the case of univariate Gaussian.
                
                Parameters:
                x : array-like
                    Data points (residuals).
                mu : float
                    Mean of the Gaussian distribution.
                sigma : float
                    Standard deviation of the Gaussian distribution.
                
                Returns:
                pdf : array-like
                    Probability density function values for each data point in x.
                """
                return univariate_gaussian_pdf(x, mu, sigma)


            # Function to compute distances from the object to the sensors
            def compute_distances(measurements, x_obj, y_obj, z_obj):
                distances = np.sqrt((measurements[:, 0] - x_obj)**2 +
                                    (measurements[:, 1] - y_obj)**2 +
                                    (measurements[:, 2] - z_obj)**2)
                return distances

            # E-step: Compute responsibilities
            def compute_responsibilities(measurements, gmm_params, object_position):
                """
                Compute the responsibilities (E-step) given the GMM parameters and object position.
                """
                x_obj, y_obj, z_obj = object_position
                predicted_ranges = compute_distances(measurements, x_obj, y_obj, z_obj)
                residuals = measurements[:, 3] - predicted_ranges
                
                # Extract GMM parameters (means, covariances, weights)
                means, covariances, weights = gmm_params['means'], gmm_params['covariances'], gmm_params['weights']
                
                # Compute likelihoods for each component
                likelihoods = np.zeros((len(residuals), len(means)))
                for k in range(len(means)):
                    likelihoods[:, k] = multivariate_gaussian_pdf(residuals, means[k], np.sqrt(covariances[k]))

                # Compute responsibilities (normalize likelihoods across components)
                weighted_likelihoods = likelihoods * weights
                total_likelihoods = np.sum(weighted_likelihoods, axis=1)
                responsibilities = weighted_likelihoods / total_likelihoods[:, np.newaxis]
                
                return responsibilities
            # M-step: Update GMM parameters and object position
            def update_gmm_and_position(measurements, responsibilities, object_position):
                """
                M-step: Update the GMM parameters and object position by maximizing the joint likelihood.
                """
                x_obj, y_obj, z_obj = object_position
                predicted_ranges = compute_distances(measurements, x_obj, y_obj, z_obj)
                residuals = measurements[:, 3] - predicted_ranges
                
                # Update GMM parameters using responsibilities
                weights = np.sum(responsibilities, axis=0) / responsibilities.shape[0]
                means = np.array([np.sum(responsibilities[:, k] * residuals) / np.sum(responsibilities[:, k])
                                for k in range(responsibilities.shape[1])])
                covariances = np.array([np.sum(responsibilities[:, k] * (residuals - means[k])**2) / np.sum(responsibilities[:, k])
                                        for k in range(responsibilities.shape[1])])
                
                # Update object position by minimizing weighted least squares
                def objective_function(position):
                    x_obj, y_obj, z_obj = position
                    predicted_ranges = compute_distances(measurements, x_obj, y_obj, z_obj)
                    residuals = measurements[:, 3] - predicted_ranges
                    weighted_residuals = np.sum(responsibilities[:, 0] * residuals**2)  # Focus on LoS (1st component)
                    return weighted_residuals
                
                initial_position = np.array(object_position)
                result = minimize(objective_function, initial_position, method='L-BFGS-B')
                
                return {'means': means, 'covariances': covariances, 'weights': weights}, result.x

            # EM algorithm for combined GMM and object position estimation
            def em_algorithm_combined(measurements, x0, y0, z0, n_components=3, max_iterations=100, tol=1e-5):
                """
                Combined EM algorithm to estimate GMM parameters and object position jointly.
                """
                # Initialize object position
                x_obj, y_obj, z_obj = x0, y0, z0
                object_position = np.array([x_obj, y_obj, z_obj])
                
                # Initialize GMM parameters (means, covariances, weights)
                gmm_params = {
                    'means': np.ones(n_components),
                    'covariances': np.array([np.eye(1) for _ in range(n_components)]),
                    'weights': np.ones(n_components) / n_components
                }
                
                for iteration in range(max_iterations):
                    # Step 1: Compute residuals
                    predicted_ranges = compute_distances(measurements, x_obj, y_obj, z_obj)
                    residuals = measurements[:, 3] - predicted_ranges
                    
                    # E-step: Compute responsibilities
                    responsibilities = compute_responsibilities(measurements, gmm_params, object_position)
                    
                    # M-step: Update both GMM parameters and object position
                    new_gmm_params, new_object_position = update_gmm_and_position(measurements, responsibilities, object_position)
                    
                    # Convergence check: based on the change in object position
                    position_change = np.linalg.norm(new_object_position - object_position)
                    if position_change < tol:
                        print(f"Converged after {iteration + 1} iterations")
                        break
                    
                    # Update parameters for next iteration
                    object_position = new_object_position
                    gmm_params = new_gmm_params
                
                return object_position, gmm_params

            x0, y0, z0 = initial_guess[:3]

            (final_x, final_y, final_z), gmm_params = em_algorithm_combined(measurements, x0, y0, z0)

            return np.array([final_x, final_y, final_z, 0, 1]), np.zeros((3,3))



    def find_closest_anchor_in_optimisation_queue(self, drone_position):
        """Function to find the closest anchor in the optimisation queue to the drone position
        
        Parameters:
        - drone_position: list of floats, the current drone position [x, y, z]
        
        Returns:
        - closest_anchor_id: int, the ID of the closest anchor in the optimisation queue
        - distance: float, the distance between the drone and the closest anchor"""
        
        closest_anchor_id = None
        distance = np.inf
        
        for anchor_id in self.anchor_to_optimise_queue.keys():
            anchor_position = self.anchor_to_optimise_queue[anchor_id]
            dist = np.linalg.norm(np.array(drone_position) - np.array(anchor_position))
            print(f"Distance between drone and anchor {anchor_id}: {dist}, the anchor is at {anchor_position}")
            if dist < distance:
                distance = dist
                closest_anchor_id = anchor_id
        anchor_ids = list(self.anchor_to_optimise_queue.keys())

        print(f"Closest anchor is {closest_anchor_id} with distance {distance}, to drone at {drone_position}")
        

        # sort by name
        sorted_anchor_ids = sorted(anchor_ids)
        #return sorted_anchor_ids[0]
        return closest_anchor_id





    def measurement_callback(self, drone_position, distance, anchor_id):
        """Callback function that is called every time a new drone position is reached, which triggers the UWB measurements
        The function processes the measurements and depending on the status of initialisation of the anchors, decides what to do with the measurements
        I.e run linear least squares, non-linear least squares, or optimise and update the trajectory
        
        Parameters:
        - drone_position: list of floats, the current drone position [x, y, z]"""

        drone_x, drone_y, drone_z = drone_position
        self.drone_position = drone_position
        
    
        # The anchor has never been seen before, process the measurement and initialise the dictionnary
        if anchor_id not in self.anchor_measurements_dictionary or self.anchor_measurements_dictionary[anchor_id]["status"] == "unseen":
            
            # If the anchor is too far, it is still possible that the status stays "unseen" and the measurement is therefore not kept
            if not self.process_measurement(drone_position, distance, anchor_id):
                return None
            # The first measurement is validated, and therefore the anchor is now seen
            else:
                self.anchor_measurements_dictionary[anchor_id]["status"] = "seen"
                return None



        # The anchor has been seen before but not enough measurements have been collected to be able to determine accurately the position and calculate an optimal trajectory
        elif self.anchor_measurements_dictionary[anchor_id]["status"] == "seen":

            if not self.process_measurement(drone_position, distance, anchor_id): # Decide what to do with the measurement, if we choose not to add it, exit and go to the other anchors
                return None
            


            anchor_measurement_dictionary = self.anchor_measurements_dictionary[anchor_id] # Extract the anchor's dictionnary 
            number_of_measurements = len(anchor_measurement_dictionary["distances_pre_rough_estimate"])

            if number_of_measurements > 10: # Threshold to start the rough estimates
                
                # Transform the measurements from the positions and range measurements to tuples
                measurements = []
                for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                    x, y, z = position
                    measurements.append([x, y, z, distance])


                ## Run the linear least squares estimation and compute the residuals
                if self.params["rough_estimate_method"] == "simple_linear":
                    estimator, covariance_matrix, residuals, x = self.estimate_anchor_position_linear_least_squares(measurements)

                elif self.params["rough_estimate_method"] == "linear_reweighted":

                    estimator, covariance_matrix, residuals, x = self.estimate_anchor_position_weighted_linear_least_squares(measurements, anchor_id)
                    new_weights = self.update_weights(residuals)

                    for _ in range(self.params['reweighting_iterations']-1): # IRLS reweighting
                        estimator, covariance_matrix, residuals, x = self.estimate_anchor_position_weighted_linear_least_squares(measurements, anchor_id, weights=new_weights)
                        new_weights = self.update_weights(residuals)

                    # Save the weights for the linear least squares
                    self.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"].append(new_weights)

                    if self.params['use_trimmed_reweighted']:
                        estimator, covariance_matrix, residuals_trimmed, x = self.estimate_anchor_position_trimmed_linear_least_squares(measurements, anchor_id, trim_fraction=0.05)


                # Compute the stopping criterion variables
                FIM = self.compute_FIM(measurements, estimator, np.diag(covariance_matrix)[:3])
                GDOP = self.compute_GDOP(measurements, estimator[:3])
                mean_residuals = np.mean(np.abs(residuals))
                condition_number = self.compute_condition_number(measurements)
                covariances = np.diag(covariance_matrix)
                verification_value = 0 # x[0]**2 + x[1]**2 + x[2]**2 - estimator[3]**2/estimator[4]**2 + x[5] # Internal residual verification, this should be zero
                distance_delta = np.linalg.norm(np.array(estimator[:3]) - np.array(anchor_measurement_dictionary["estimator_rough_linear"][:3])) # Distance between two consecutive estimates

                # Add them to the dictionnary for plotting
                anchor_measurement_dictionary["FIM"].append(FIM)
                anchor_measurement_dictionary["GDOP"].append(GDOP)
                anchor_measurement_dictionary["residuals"].append(mean_residuals)
                anchor_measurement_dictionary["condition_number"].append(condition_number)
                anchor_measurement_dictionary["covariances"].append(covariances)
                anchor_measurement_dictionary["verification_vector"].append(np.abs(verification_value))
                anchor_measurement_dictionary["residual_vector"].append(residuals)
                anchor_measurement_dictionary["consecutive_distances_vector"].append(distance_delta)

                # Update the estimated position of the anchor
                anchor_measurement_dictionary["estimator_rough_linear"] = estimator
                anchor_measurement_dictionary["covariance_matrix_rough_linear"] = covariance_matrix
                anchor_measurement_dictionary["estimator"] = estimator
                anchor_measurement_dictionary["covariance_matrix"] = covariance_matrix
                
                # Outlier removal part, using the residuals
                if self.params["outlier_removing"] != "None":
                    if self.outlier_filtering(anchor_measurement_dictionary, residuals, self.params["outlier_removing"]): 
                        # If we just removed an outlier, it makes sense to wait an iteration before using the stopping criterion, to avoid the effect of the outlier
                        return None
                


                # Check if the stopping criterion is achieved based on the variables calculated above
                if self.stopping_criterion_check(number_of_measurements, anchor_measurement_dictionary, choose_criterion=self.params["stopping_criteria"]): # choose_criterion=["condition_number","consecutive_distances_vector"]):
                    print("Stopping criterion achieved for anchor ", anchor_id)
                    # If the stopping criterion is achieved, we can now refine the rough estimate with a non-linear least squares

                    self.drone_position = drone_position

                    # Refine the linear rough estimate with the non-linear rough estimate using the linear one as an intial guess
                    estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_linear"])

                    # Update the dictionnary with the non-linear refined rough estimate
                    anchor_measurement_dictionary["estimator_rough_non_linear"] = estimator
                    anchor_measurement_dictionary["estimator"] = estimator # estimator contains the most accurate estimate (so the non linear one)
                    anchor_estimate_variance = anchor_measurement_dictionary["covariance_matrix_rough_linear"][:3]

                    # We now have an accurate initial estimate of the anchor, we can start the trajectory optimisation ONLY if the drone is currently not on another optimal trajectory
                    if self.status == "on_optimal_trajectory":
                        anchor_measurement_dictionary["status"] = "stopping_criterion_triggered" # Set the status to stopping criterion triggered to keep in memory that the stopping criterion was achieved at some point
                        self.anchor_to_optimise_queue[anchor_id] = anchor_measurement_dictionary["estimator"][:3] # Add the anchor to the optimisation queue, meaning it is ready to be optimised for an optimal trajectory
                        return None
                    
                    # If the drone is not on an optimal trajectory, we can start the trajectory optimisation

                    previous_measurement_positions = anchor_measurement_dictionary["positions_pre_rough_estimate"]

                    initial_remaining_waypoints = self.remaining_waypoints
                    print("Initial remaining waypoints: ", initial_remaining_waypoints)

                    self.trajectory_optimiser.method = self.params["trajectory_optimisation_method"]
                    link_method = self.params["link_method"]

                    # Optimize the trajectory using the previous measurements and the rough estimate of the anchor 
                    optimal_waypoints = self.trajectory_optimiser.optimize_waypoints_incrementally_spherical(drone_position, estimator, anchor_estimate_variance, previous_measurement_positions, initial_remaining_waypoints, radius_of_search = 0.2, max_waypoints=20, marginal_gain_threshold=0.01)
                    
                    return_waypoints = None
                    if link_method == "optimal":
                        optimal_waypoints, return_waypoints = self.trajectory_optimiser.optimize_return_waypoints_incrementally_spherical(optimal_waypoints[-1], estimator, anchor_estimate_variance, previous_measurement_positions, initial_remaining_waypoints[0], radius_of_search = 0.2, max_waypoints=20, marginal_gain_threshold=0.01, lambda_penalty=1)
                    
                    full_waypoints, optimal_waypoints, link_waypoints, mission_waypoints = self.trajectory_optimiser.compute_new_mission_waypoints(self.passed_waypoints[-1], initial_remaining_waypoints, optimal_waypoints, link_method, return_waypoints)
                    

                    if len(optimal_waypoints) > 0:
                        # full_waypoints, optimal_waypoints, link_waypoints, mission_waypoints = self.trajectory_optimiser.compute_new_mission_waypoints(self.passed_waypoints[-1], initial_remaining_waypoints, optimal_waypoints, link_method)
                         # Update the status of the anchor to optimised trajectory
                        self.anchor_measurements_dictionary[anchor_id]["status"] = "optimised_trajectory"

                        # Update the status of the drone to on optimal trajectory
                        self.status = "on_optimal_trajectory"

                    else:
                        full_waypoints = initial_remaining_waypoints
                        optimal_waypoints = []
                        link_waypoints = []
                        mission_waypoints = initial_remaining_waypoints
                        self.anchor_measurements_dictionary[anchor_id]["status"] = "initialised"
                    
                   

                    self.current_optimal_waypoints = optimal_waypoints
                    self.current_link_waypoints = link_waypoints
                    self.remaining_waypoints = full_waypoints # mission_waypoints

                    return full_waypoints




            else: # Number of measurements not enough to perform the stopping criterion analysis

                # We still append some values to the dictionnary to keep track of the progress and align them with the measurement count

                FIM = np.zeros((5,5))
                GDOP = float('inf')
                residuals = float('inf') # Mean of the residuals
                condition_number = float('inf')
                covariances = [float('inf'),float('inf'),float('inf')]
                verification_value = float('inf')
                weight = [] # empty for now
                delta_estimate = float('inf')

                anchor_measurement_dictionary["FIM"].append(FIM)
                anchor_measurement_dictionary["GDOP"].append(GDOP)
                anchor_measurement_dictionary["residuals"].append(residuals)
                anchor_measurement_dictionary["condition_number"].append(condition_number)
                anchor_measurement_dictionary["covariances"].append(covariances)
                anchor_measurement_dictionary["verification_vector"].append(verification_value)

                anchor_measurement_dictionary["linear_ls_weights"].append(weight)
                anchor_measurement_dictionary["residual_vector"].append(weight)
                anchor_measurement_dictionary["consecutive_distances_vector"].append(delta_estimate)
                return
        

        # We get a new measurement for an anchor that already has good enough measurements and would be ready to be optimised
        elif self.anchor_measurements_dictionary[anchor_id]["status"] == "stopping_criterion_triggered":
            
            # If the drone is still on the optimal trajectory, we can continue collecting measurements, but we do not need to do anything with them
            if self.status == "on_optimal_trajectory":
                self.process_measurement(drone_position, distance, anchor_id) # Decide what to do with the measurement, if we choose not to add it, exit and go to the other anchors
                return None


            # If the drone is not on the optimal trajectory anymore, we use the closest anchor in the optimisation queue to start the optimisation
            elif self.status != "on_optimal_trajectory":
                
                self.process_measurement(drone_position, distance, anchor_id) # Add the measurement to the dictionnary if eligible

                self.drone_position = drone_position # Save the drone position at which the rough estimate was calculated

                if anchor_id != self.find_closest_anchor_in_optimisation_queue(drone_position):
                    # If this anchor is not the closest, exit as we would prefer to optimise for the closest anchor
                    return None
                
                # If the anchor is in fact the closest, we can start the optimisation process

                # First rerun the linear and non linear least squares to get the most accurate estimate

                anchor_measurement_dictionary = self.anchor_measurements_dictionary[anchor_id] # Extract the anchor's dictionnary 
                
                # Transform the measurements from the positions and range measurements to tuples
                measurements = []
                for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                    x, y, z = position
                    measurements.append([x, y, z, distance])

                ## Run the linear least squares estimation and compute the residuals
                if self.params["rough_estimate_method"] == "simple_linear":
                    estimator, covariance_matrix, residuals, x = self.estimate_anchor_position_linear_least_squares(measurements)

                elif self.params["rough_estimate_method"] == "linear_reweighted":

                    estimator, covariance_matrix, residuals, x = self.estimate_anchor_position_weighted_linear_least_squares(measurements, anchor_id)
                    new_weights = self.update_weights(residuals)

                    for _ in range(self.params['reweighting_iterations']-1): # IRLS reweighting
                        estimator, covariance_matrix, residuals, x = self.estimate_anchor_position_weighted_linear_least_squares(measurements, anchor_id, weights=new_weights)
                        new_weights = self.update_weights(residuals)

                    # Save the weights for the linear least squares
                    self.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"].append(new_weights)

                    if self.params['use_trimmed_reweighted']:
                        estimator, covariance_matrix, residuals_trimmed, x = self.estimate_anchor_position_trimmed_linear_least_squares(measurements, anchor_id, trim_fraction=0.05)
                
                anchor_measurement_dictionary["estimator_rough_linear"] = estimator
                anchor_measurement_dictionary["covariance_matrix_rough_linear"] = covariance_matrix

                # Refine the linear rough estimate with the non-linear rough estimate using the linear one as an intial guess
                estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_linear"])

                # Update the dictionnary with the non-linear refined rough estimate
                anchor_measurement_dictionary["estimator_rough_non_linear"] = estimator
                anchor_measurement_dictionary["estimator"] = estimator

                anchor_estimate = estimator[:3]
                anchor_estimate_variance = anchor_measurement_dictionary["covariance_matrix_rough_linear"][:3]

                previous_measurement_positions = anchor_measurement_dictionary["positions_pre_rough_estimate"]

                # TODO:
                # Decide wether or not to return to the link waypoint or not depending on the geometry and the distance to the anchor
                if len(self.current_link_waypoints) == 0:
                    optimal_trajectory_starting_point = drone_position
                    print(f"Optimising for anchor {anchor_id} with starting point {optimal_trajectory_starting_point}, which is the drone position")

                    # If the drone is closer to the the anchor estimate than the link waypoint is, we start from link waypoint
                elif np.linalg.norm(np.array(anchor_estimate) - np.array(self.current_link_waypoints[-1])) < np.linalg.norm(np.array(drone_position) - np.array(anchor_estimate)):
                    optimal_trajectory_starting_point = self.current_link_waypoints[-1]
                    print(f"Optimising for anchor {anchor_id} with starting point {optimal_trajectory_starting_point}, which is the link waypoint")
                else:
                    optimal_trajectory_starting_point = drone_position
                    print(f"Optimising for anchor {anchor_id} with starting point {optimal_trajectory_starting_point}, which is the drone position")
                
                initial_remaining_waypoints = self.remaining_waypoints
                
                self.trajectory_optimiser.method = self.params["trajectory_optimisation_method"]
                link_method = self.params["link_method"]
                
                # Optimize the trajectory using the previous measurements and the rough estimate of the anchor 
                optimal_waypoints = self.trajectory_optimiser.optimize_waypoints_incrementally_spherical(optimal_trajectory_starting_point, estimator, anchor_estimate_variance, previous_measurement_positions, initial_remaining_waypoints, radius_of_search = 0.2, max_waypoints=20, marginal_gain_threshold=0.01)
                return_waypoints = None
                if link_method == "optimal":
                    optimal_waypoints, return_waypoints = self.trajectory_optimiser.optimize_return_waypoints_incrementally_spherical(optimal_waypoints[-1], estimator, anchor_estimate_variance, previous_measurement_positions, initial_remaining_waypoints[0], radius_of_search = 0.2, max_waypoints=20, marginal_gain_threshold=0.01, lambda_penalty=1)
                full_waypoints, optimal_waypoints, link_waypoints, mission_waypoints = self.trajectory_optimiser.compute_new_mission_waypoints(self.passed_waypoints[-1], initial_remaining_waypoints, optimal_waypoints, link_method, return_waypoints)
                
                
                # Update the status of the anchor to optimised trajectory
                self.anchor_measurements_dictionary[anchor_id]["status"] = "optimised_trajectory"

                # Update the status of the drone to on optimal trajectory
                self.status = "on_optimal_trajectory"
                if return_waypoints is not None:
                    self.current_optimal_waypoints = optimal_waypoints + return_waypoints
                else:
                    self.current_optimal_waypoints = optimal_waypoints

                self.current_link_waypoints = link_waypoints
                self.remaining_waypoints = full_waypoints # mission_waypoints
                

                return full_waypoints

        # The anchor has been seen before and enough measurements have been collected to perform the stopping criterion analysis, We are now in the optimised trajectory phase for this anchor
        elif self.anchor_measurements_dictionary[anchor_id]["status"] == "optimised_trajectory":


            # extract the anchor's dictionnary
            anchor_measurement_dictionary = self.anchor_measurements_dictionary[anchor_id]

            if len(self.current_optimal_waypoints) == 0:
                
                measurements = []
                for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                    x, y, z = position
                    measurements.append([x, y, z, distance])

                estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_non_linear"])

                anchor_measurement_dictionary["estimator"] = estimator
                anchor_measurement_dictionary["covariance_matrix"] = covariance_matrix

                anchor_measurement_dictionary["status"] = "initialised"
                print(f"Anchor {anchor_id} is now initialised")
                # Update the status of the drone to open to measurements
                self.status = "open_to_measurements"

                # Remove the anchor from the optimisation queue, to leave space for other anchors
                if anchor_id in self.anchor_to_optimise_queue.keys():
                    del self.anchor_to_optimise_queue[anchor_id]
                self.current_optimal_waypoints = []
                return None
            
            # process the measurement on the optimal trajectory
            self.process_measurement_optimal_trajectory(drone_position, distance, anchor_id) # Decide what to do with the measurement

            if (np.array(self.passed_waypoints[-1] == self.current_optimal_waypoints[-1])).all() and len(self.current_optimal_waypoints) < 2:
            # If all measurements on the optimal trajectory are collected, run the final estimate and set it as initialised
            #if np.linalg.norm(np.array(drone_position) - np.array([self.current_optimal_waypoints[-1][0], self.current_optimal_waypoints[-1][1], self.current_optimal_waypoints[-1][2]])) < 0.01:
            #if np.linalg.norm(np.array(drone_position) - np.array([self.current_link_waypoints[-1][0], self.current_link_waypoints[-1][1], self.current_link_waypoints[-1][2]])) < 0.001:

                measurements = []
                for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                    x, y, z = position
                    measurements.append([x, y, z, distance])

                estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_non_linear"])

                anchor_measurement_dictionary["estimator"] = estimator
                anchor_measurement_dictionary["covariance_matrix"] = covariance_matrix

                # Set the anchor status to initialised
                anchor_measurement_dictionary["status"] = "initialised"
                print(f"Anchor {anchor_id} is now initialised")
                # Update the status of the drone to open to measurements
                self.status = "open_to_measurements"

                # Remove the anchor from the optimisation queue, to leave space for other anchors
                if anchor_id in self.anchor_to_optimise_queue.keys():
                    del self.anchor_to_optimise_queue[anchor_id]

                
            
            return None
        
        elif self.anchor_measurements_dictionary[anchor_id]["status"] == "initialised":
            return None
        
        # Not sure what could be here, but enables the error to be caught
        else:
            print(f"WARNING: Anchor status not recognised for anchor {anchor_id}")
            return None


    def process_measurement(self, drone_position, distance, anchor_id):
        """Process a new measurement and decide whether or not to add it to the dictionnary of measurements for the anchor
        
        Parameters:
        - drone_position: list of floats, the current drone position [x, y, z]
        - distance: float, the measured distance to the anchor
        - anchor_id: int, the ID of the anchor
        
        Returns:
        - bool, True if the measurement was added, False if it was not added
        """
        
        drone_x, drone_y, drone_z = drone_position

        anchor_measurement_dictionary = self.anchor_measurements_dictionary.setdefault(anchor_id, deepcopy(self.default_anchor_structure))

        
        # Reject measurements that are too far away
        if distance > self.params["distance_rejection_threshold"]:
            return False # We did not add a measurement
        
        
        # I we have no measurements yet, or if we decide to gather mutiple measurements at points very close to each other for robustness, add the measurement to the dictionnary
        if (len(anchor_measurement_dictionary["distances_pre_rough_estimate"]) % self.params["number_of_redundant_measurements"] != 0) or len(anchor_measurement_dictionary["distances_pre_rough_estimate"])  == 0:

            anchor_measurement_dictionary["distances_pre_rough_estimate"].append(distance)
            anchor_measurement_dictionary["positions_pre_rough_estimate"].append((drone_x, drone_y, drone_z))

        # If the drone has moved significantly since the last measurement, add the new measurement to the dictionnary
        elif np.linalg.norm(np.array(anchor_measurement_dictionary["positions_pre_rough_estimate"][-1]) - np.array((drone_x, drone_y, drone_z))) / distance > self.params["distance_to_anchor_ratio_threshold"]:
            anchor_measurement_dictionary["distances_pre_rough_estimate"].append(distance)
            anchor_measurement_dictionary["positions_pre_rough_estimate"].append((drone_x, drone_y, drone_z))

        else: 
            return False # No measurement was added
        
        return True # A measurement was added
    
    def process_measurement_optimal_trajectory(self, drone_position, distance, anchor_id):
        """Process a new measurement and decide whether or not to add it to the dictionnary of measurements for the anchor
        
        Parameters:
        - drone_position: list of floats, the current drone position [x, y, z]
        - distance: float, the measured distance to the anchor
        - anchor_id: int, the ID of the anchor
        
        Returns:
        - bool, True if the measurement was added, False if it was not added
        """
        
        drone_x, drone_y, drone_z = drone_position

        anchor_measurement_dictionary = self.anchor_measurements_dictionary.setdefault(anchor_id, deepcopy(self.default_anchor_structure))

        # Reject measurements that are too far away
        if distance > self.params["distance_rejection_threshold"]:
            return False # We did not add a measurement
        
        
        # I we have no measurements yet, or if we decide to gather mutiple measurements at points very close to each other for robustness, add the measurement to the dictionnary
        if (len(anchor_measurement_dictionary["distances_post_rough_estimate"]) % self.params["number_of_redundant_measurements"] != 0):

            anchor_measurement_dictionary["distances_post_rough_estimate"].append(distance)
            anchor_measurement_dictionary["positions_post_rough_estimate"].append((drone_x, drone_y, drone_z))

        # If the drone has moved significantly since the last measurement, add the new measurement to the dictionnary
        elif len(anchor_measurement_dictionary["distances_post_rough_estimate"])  == 0 and np.linalg.norm(np.array(anchor_measurement_dictionary["positions_pre_rough_estimate"][-1]) - np.array((drone_x, drone_y, drone_z))) / distance > self.params["distance_to_anchor_ratio_threshold"]:
            anchor_measurement_dictionary["distances_post_rough_estimate"].append(distance)
            anchor_measurement_dictionary["positions_post_rough_estimate"].append((drone_x, drone_y, drone_z))

        elif len(anchor_measurement_dictionary["distances_post_rough_estimate"])  != 0 and np.linalg.norm(np.array(anchor_measurement_dictionary["positions_post_rough_estimate"][-1]) - np.array((drone_x, drone_y, drone_z))) / distance > self.params["distance_to_anchor_ratio_threshold"]:
            anchor_measurement_dictionary["distances_post_rough_estimate"].append(distance)
            anchor_measurement_dictionary["positions_post_rough_estimate"].append((drone_x, drone_y, drone_z))

        else: 
            return False # No measurement was added
        
        return True # A measurement was added
    
    def stopping_criterion_check(self, number_of_measurements, anchor_meas_dictionnary, choose_criterion=None):
        """Check if the stopping criterion is met based on the variables computed during the initialisation process to decide when to stop collecting measurements

        Parameters:
        - number_of_measurements: int, the number of measurements collected for the anchor
        - anchor_meas_dictionnary: dict, the dictionnary containing the measurements for the anchor
        
        Returns:
        - bool, True if the stopping criterion is met, False if it is not met
        """

        def check_criterion_convergence(criterion_array, marginal_threshold):
            """Check if the criterion has converged based on the marginal threshold
            
            Parameters:
            - criterion_array: numpy array, the array of the criterion values
            - marginal_threshold: float, the threshold for the marginal gain
            
            Returns:
            - bool, True if the criterion has converged, False if it has not
            """

            if len(criterion_array) < 2:
                return False

            criterion_k_prev, criterion_k = criterion_array[-2], criterion_array[-1]
            criterion_ratio = np.absolute(criterion_k - criterion_k_prev) / criterion_k_prev if criterion_k_prev != 0 and not np.isinf(criterion_k_prev) else float('inf')

            if criterion_ratio < marginal_threshold:
                return True
            else:
                return False
            
        def check_criterion_threshold(criterion_array, threshold):
            """Check if the criterion has reached the threshold
            
            Parameters:
            - criterion_array: numpy array, the array of the criterion values
            - threshold: float, the threshold for the criterion
            
            Returns:
            - bool, True if the criterion has reached the threshold, False if it has not
            """

            if len(criterion_array) < 1:
                return False

            criterion_k = criterion_array[-1]

            if criterion_k < threshold:
                return True
            else:
                return False
            
        if choose_criterion is None:
            return False

        if "nb_measurements" in choose_criterion:
            if number_of_measurements > self.params["number_of_measurements_thresh"]:
                return True
            if choose_criterion == ["nb_measurements"]:
                return False
            
        
        if "GDOP" in choose_criterion:
            if not check_criterion_convergence(anchor_meas_dictionnary["GDOP"], self.params["GDOP_ratio_thresh"]):
                anchor_meas_dictionnary["GDOP_convergence_counter"] = 0
                return False
            
            anchor_meas_dictionnary["GDOP_convergence_counter"] += 1

            if not anchor_meas_dictionnary["GDOP_convergence_counter"] > self.params["convergence_counter_threshold"]:
                return False
            if not check_criterion_threshold(anchor_meas_dictionnary["GDOP"], self.params["GDOP_thresh"]):
                return False
            
            
        if "FIM" in choose_criterion:

            fim_inverse_determinants = [1/(np.linalg.det(fim)) for fim in anchor_meas_dictionnary["FIM"][-2:]]
            if not check_criterion_convergence(fim_inverse_determinants, self.params["FIM_ratio_thresh"]):
                anchor_meas_dictionnary["FIM_convergence_counter"] = 0
                return False
            anchor_meas_dictionnary["FIM_convergence_counter"] += 1
            if not anchor_meas_dictionnary["FIM_convergence_counter"] > self.params["convergence_counter_threshold"]:
                return False
            if not check_criterion_threshold(fim_inverse_determinants, self.params["FIM_thresh"]):
                return False
            
        if "residuals" in choose_criterion:
            if not check_criterion_convergence(anchor_meas_dictionnary["residuals"], self.params["residuals_ratio_thresh"]):
                anchor_meas_dictionnary["residuals_convergence_counter"] = 0
                return False
            anchor_meas_dictionnary["residuals_convergence_counter"] += 1
            if not anchor_meas_dictionnary["residuals_convergence_counter"] > self.params["convergence_counter_threshold"]:
                return False
            if not check_criterion_threshold(anchor_meas_dictionnary["residuals"], self.params["residuals_thresh"]):
                return False
        
        if "covariances" in choose_criterion:
            covariances = [np.max(cov[:3]) for cov in anchor_meas_dictionnary["covariances"]]
            if not check_criterion_convergence(covariances, self.params["covariance_ratio_thresh"]):
                anchor_meas_dictionnary["covariances_convergence_counter"] = 0
                return False
            anchor_meas_dictionnary["covariances_convergence_counter"] += 1
            if not anchor_meas_dictionnary["covariances_convergence_counter"] > self.params["convergence_counter_threshold"]:
                return False
            if not check_criterion_threshold(covariances, self.params["covariance_thresh"]):
                return False


        if "condition_number" in choose_criterion:
            if not check_criterion_convergence(anchor_meas_dictionnary["condition_number"], self.params["condition_number_ratio_thresh"]):
                anchor_meas_dictionnary["condition_number_convergence_counter"] = 0
                return False
            anchor_meas_dictionnary["condition_number_convergence_counter"] += 1
            if not anchor_meas_dictionnary["condition_number_convergence_counter"] > self.params["convergence_counter_threshold"]:
                return False
            if not check_criterion_threshold(anchor_meas_dictionnary["condition_number"], self.params["condition_number_thresh"]):
                return False
            
        if "verification_vector" in choose_criterion:
            if not check_criterion_convergence(anchor_meas_dictionnary["verification_vector"], self.params["verification_vector_ratio_thresh"]):
                anchor_meas_dictionnary["verification_vector_convergence_counter"] = 0
                return False
            anchor_meas_dictionnary["verification_vector_convergence_counter"] += 1
            if not anchor_meas_dictionnary["verification_vector_convergence_counter"] > self.params["convergence_counter_threshold"]:
                return False
            if not check_criterion_threshold(anchor_meas_dictionnary["verification_vector"], self.params["verification_vector_thresh"]):
                return False
            
        if "consecutive_distances_vector" in choose_criterion:

            if anchor_meas_dictionnary["consecutive_distances_vector"][-1] < self.params["convergence_postion_thresh"]:
                anchor_meas_dictionnary["consecutive_distances_vector_convergence_counter"] += 1
            else:
                anchor_meas_dictionnary["consecutive_distances_vector_convergence_counter"] = 0

            if not anchor_meas_dictionnary["consecutive_distances_vector_convergence_counter"] > self.params["convergence_counter_threshold"]:
                return False


        return True

    



    def outlier_finder(self, residuals):
        """Find the outliers in the residuals using the z-score method
        
        Parameters:
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - outliers: numpy array, the indices of the outliers in the residuals
        """
        len_measurements = len(residuals)

        threshold = self.params["z_score_threshold"]

        # if len_measurements < 10:
        #     threshold = 5*threshold
        # elif len_measurements < 25:
        #     threshold = 3.5*threshold
        # elif len_measurements < 50:
        #     threshold = 2.5*threshold
        # elif len_measurements < 75:
        #     threshold = 1.5*threshold
        # else:
        #     threshold = 1*threshold

        z_score = self.compute_z_score(residuals)

        outliers = np.where(np.abs(z_score) > threshold)[0]
        return outliers

    # def cluster_measurements(self, measurements, number_of_clusters=15):
    #     """Cluster the measurements using the KMeans algorithm

    #     Parameters:
    #     - measurements: numpy array, the measurements to cluster
    #     - number_of_clusters: int, the number of clusters to use for the KMeans algorithm

    #     Returns:
    #     - centroids: numpy array, the centroids of the clusters
    #     """

    #     kmeans = KMeans(n_clusters=number_of_clusters)
    #     kmeans.fit_predict(measurements)
    #     centroids = kmeans.cluster_centers_

    #     return centroids

    def outlier_filtering(self, anchor_measurement_dictionary, residuals, method):
        """Filter the outliers in the measurements using the z-score method
        
        Parameters:
        - anchor_measurement_dictionary: dict, the dictionnary containing the measurements for the anchor
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - bool, True if outliers were found and removed, False if no outliers were found
        """
        
        outliers = self.outlier_finder(residuals)

        if method == "immediate":
            anchor_measurement_dictionary["distances_pre_rough_estimate"] = np.delete(anchor_measurement_dictionary["distances_pre_rough_estimate"], outliers).tolist()
            anchor_measurement_dictionary["positions_pre_rough_estimate"] = np.delete(anchor_measurement_dictionary["positions_pre_rough_estimate"], outliers, axis=0).tolist()

            if self.params["rough_estimate_method"] == "linear_reweighted":
                anchor_measurement_dictionary["linear_ls_weights"][-1] = np.delete(anchor_measurement_dictionary["linear_ls_weights"][-1], outliers, axis=0).tolist()

            if outliers.size > 0:

                return True
            else:
                return False
            
        elif method == "counter":

            number_of_measurements = len(anchor_measurement_dictionary["distances_pre_rough_estimate"])
            outlier_counter_length = len(anchor_measurement_dictionary["linear_ls_outlier_counter"])

            if outlier_counter_length < number_of_measurements:
                anchor_measurement_dictionary["linear_ls_outlier_counter"] = np.append(anchor_measurement_dictionary["linear_ls_outlier_counter"], np.zeros(number_of_measurements - outlier_counter_length))

            mask = np.zeros_like(anchor_measurement_dictionary["linear_ls_outlier_counter"])
            mask[outliers] = 1

            anchor_measurement_dictionary["linear_ls_outlier_counter"] += np.ones_like(anchor_measurement_dictionary["linear_ls_outlier_counter"])
            anchor_measurement_dictionary["linear_ls_outlier_counter"] *= mask

            for outlier_id, outlier_count in enumerate(anchor_measurement_dictionary["linear_ls_outlier_counter"]):
                if outlier_count > self.params["outlier_count_threshold"]:
                    anchor_measurement_dictionary["distances_pre_rough_estimate"].pop(outlier_id)
                    anchor_measurement_dictionary["positions_pre_rough_estimate"].pop(outlier_id)
                    if self.params["rough_estimate_method"] == "linear_reweighted":
                        anchor_measurement_dictionary["linear_ls_weights"][-1].pop(outlier_id)


            return False # We suppose the method used is quite robust and therefore we do not need to remove the outliers immediately and can still compute the stopping criterion

        
    



    def calculate_position_error(self, real_position, estimated_position):
            """Calculate the error between the estimated anchor position and the real ground truth anchor position
            Can be used for one or multiple anchors at once
            
            Parameters:
            - real_positions: numpy array, the real ground truth positions of the anchors
            - estimated_positions: numpy array, the estimated positions of the anchors
            
            Returns:
            - error: float, the error between the estimated and real positions
            """
            real_position = np.array(real_position)
            estimated_position = np.array(estimated_position)

            error = np.linalg.norm(real_position - estimated_position)
            return error
    
    def calculate_estimator_error(self, real_anchor_values, estimator):
        """Calculate the error between the estimated anchor estimator and the real ground truth anchor characteristics
        Can be used for one or multiple anchors at once
        The estimator contains the position of the anchor and the bias and linear bias
        
        Parameters:
        - real_anchor_values: numpy array, the real ground truth values of the anchors
        - estimator: numpy array, the estimated values of the anchors
        """
        
        real_anchor_values = np.array(real_anchor_values)
        estimator = np.array(estimator)

        error = np.linalg.norm(estimator - real_anchor_values)

        return error
    
    def save_measurements_to_csv(self, anchor_id, filename):
        """Save the measurements of an anchor to a csv file
        
        Parameters:
        - anchor_id: int, the ID of the anchor
        - filename: string, the name of the file to save the measurements to
        """
        
        anchor_measurement_dictionary = self.anchor_measurements_dictionary[anchor_id]
            
        
        distances_pre_rough_estimate = anchor_measurement_dictionary["distances_pre_rough_estimate"]
        positions_pre_rough_estimate = anchor_measurement_dictionary["positions_pre_rough_estimate"]

        distances_post_rough_estimate = anchor_measurement_dictionary["distances_post_rough_estimate"]
        positions_post_rough_estimate = anchor_measurement_dictionary["positions_post_rough_estimate"]
        
        linear_estimate = anchor_measurement_dictionary["estimator_rough_linear"]
        non_linear_esstimate = anchor_measurement_dictionary["estimator_rough_non_linear"]
        final_estimate = anchor_measurement_dictionary["estimator"]

        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(["anchor_id", "distance", "x", "y", "z"])
            for distance, position in zip(distances_pre_rough_estimate, positions_pre_rough_estimate):
                x, y, z = position
                writer.writerow([anchor_id, distance, x, y, z])

            # Add a blank line to separate the pre and post rough estimate measurements
            writer.writerow([])

            writer.writerow(linear_estimate)

            writer.writerow([])

            writer.writerow(non_linear_esstimate)

            writer.writerow([])

            for distance, position in zip(distances_post_rough_estimate, positions_post_rough_estimate):
                x, y, z = position
                writer.writerow([anchor_id, distance, x, y, z])

            writer.writerow([])

            writer.writerow(final_estimate)
            






    ### Helper functions for quickly computing the anchor initialisation without relying on the simulator to run real time
    
    def reset_measurements_post_rough_initialisation(self, anchor_id):
        """Reset the measurements in the dictionnary that were gathered after the rough estimation of the anchor from the optimal waypoints
        
        Parameters:
        - anchor_id: int, the ID of the anchor
        """
        self.anchor_measurements_dictionary[anchor_id]["status"] = "optimised_trajectory"
        self.anchor_measurements_dictionary[anchor_id]["distances_post_rough_estimate"] = []
        self.anchor_measurements_dictionary[anchor_id]["positions_post_rough_estimate"] = []
        self.anchor_measurements_dictionary[anchor_id]["estimator"] = self.anchor_measurements_dictionary[anchor_id]["estimator_rough_non_linear"]
        self.anchor_measurements_dictionary[anchor_id]["covariance_matrix"] = []
        
    def reset_all_measurements(self, anchor_id):
        """Reset all the measurements in the dictionary for a specific anchor
        
        Parameters:
        - anchor_id: int, the ID of the anchor"""

        self.anchor_measurements_dictionary[anchor_id] = deepcopy(self.default_anchor_structure)
    
