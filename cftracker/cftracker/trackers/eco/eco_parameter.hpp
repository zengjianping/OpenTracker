// Set the value the same as testing_ECO_gpu.m
#ifndef CFTRACKER_ECO_PARAMETERS_HPP
#define CFTRACKER_ECO_PARAMETERS_HPP

#include "cftracker/feature/feature_parameter.hpp"
#include "cftracker/filters/scale_parameter.hpp"


namespace cftracker {

// Cojugate Gradient Options Structure
struct CgOpts {
	bool debug;
	bool CG_use_FR;
	float tol;
	bool CG_standard_alpha;
	float init_forget_factor;
	int maxit;
};

// Parameters set exactly the same as 'testing_ECO_HC.m'
struct EcoParameters {
	FeatureParameters feature_params;

	// Scale filter parameters
	// Only used if: use_scale_filter = true
	bool use_scale_filter = false; // Use the fDSST scale filter or not (for speed)
	ScaleParameters scale_params;
	
	// extra parameters
	CgOpts CG_opts;
	float max_score_threshhold = 0.1;

	// global feature parameters1s
	int normalize_power = 2;
	bool normalize_size = true;
	bool normalize_dim = true;

	// img sample parameters
	std::string search_area_shape = "square"; // The shape of the samples
	float search_area_scale = 4.0;		 // The scaling of the target size to get the search area
	int min_image_sample_size = 22500;   // Minimum area of image samples, 200x200
	int max_image_sample_size = 40000;   // Maximum area of image samples, 250x250

	// detection parameters
	int refinement_iterations = 1; // Number of iterations used to refine the resulting position in a frame
	int newton_iterations = 5;	 // The number of Newton iterations used for optimizing the detection score
	bool clamp_position = false;   // Clamp the target position to be inside the image

	// learning parameters
	float output_sigma_factor = 1.0f / 16.0f; // Label function sigma
	float learning_rate = 0.009; // Learning rate
	size_t nSamples = 30; // Maximum number of stored training samples
	std::string sample_replace_strategy = "lowest_prior"; // Which sample to replace when the memory is full
	bool lt_size = 0; // The size of the long - term memory(where all samples have equal weight)
	int train_gap = 5; // The number of intermediate frames with no training(0 corresponds to training every frame)
	int skip_after_frame = 10; // After which frame number the sparse update scheme should start(1 is directly)
	bool use_detection_sample = true; // Use the sample that was extracted at the detection stage also for learning

	// factorized convolution parameters
	bool use_projection_matrix = true;	// Use projection matrix, i.e. use the factorized convolution formulation
	bool update_projection_matrix = true; // Whether the projection matrix should be optimized or not
	std::string proj_init_method = "pca"; // Method for initializing the projection matrix
	float projection_reg = 1e-7; // Regularization paremeter of the projection matrix (lambda)

	// Generative sample space model parameters
	bool use_sample_merge = true; // Use the generative sample space model to merge samples
	std::string sample_merge_type = "Merge"; // Strategy for updating the samples
	std::string distance_matrix_update_type = "exact"; // Strategy for updating the distance matrix

	// Conjugate Gradient parameters
	int CG_iter = 5; // The number of Conjugate Gradient iterations in each update after the first frame
	int init_CG_iter = 10 * 15; // The total number of Conjugate Gradient iterations used in the first frame
	int init_GN_iter = 10; // The number of Gauss-Newton iterations used in the first frame(only if the projection matrix is updated)
	bool CG_use_FR = false; // Use the Fletcher-Reeves(true) or Polak-Ribiere(false) formula in the Conjugate Gradient
	bool CG_standard_alpha = true;  // Use the standard formula for computing the step length in Conjugate Gradient
	int CG_forgetting_rate = 50; // Forgetting rate of the last conjugate direction
	float precond_data_param = 0.75; // Weight of the data term in the preconditioner
	float precond_reg_param = 0.25; // Weight of the regularization term in the preconditioner
	int precond_proj_param = 40; // Weight of the projection matrix part in the preconditioner

	// regularization window parameters
	bool use_reg_window = true; // Use spatial regularization or not
	double reg_window_min = 1e-4; // The minimum value of the regularization window
	double reg_window_edge = 10e-3; // The impact of the spatial regularization
	size_t reg_window_power = 2; // The degree of the polynomial to use(e.g. 2 is a quadratic window)
	float reg_sparsity_threshold = 0.05; // A relative threshold of which DFT coefficients that should be set to zero

	// Interpolation parameters
	std::string interpolation_method = "bicubic"; // The kind of interpolation kernel
	float interpolation_bicubic_a = -0.75;   // The parameter for the bicubic interpolation kernel
	bool interpolation_centering = true; // Center the kernel at the feature sample
	bool interpolation_windowing = false; // Do additional windowing on the Fourier coefficients of the kernel

	// Scale parameters for the translation model
	// Only used if: use_scale_filter = false
	size_t number_of_scales = 7; // Number of scales to run the detector
	float scale_step = 1.01f; // The scale factor
	float min_scale_factor;
	float max_scale_factor;

	bool debug = false; // to show heatmap or not

	// GPU
	bool use_gpu = true; // whether Caffe use gpu or not
	int gpu_id = 0;
};

} // namespace cftracker

#endif // CFTRACKER_ECO_PARAMETERS_HPP


