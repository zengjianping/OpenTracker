#ifndef SCALE_FILTER_HPP
#define SCALE_FILTER_HPP

#include <opencv2/core.hpp>

namespace cftracker {

struct ScaleFilterParameter {
	float scale_sigma_factor = 1.0f / 16.0f; // Scale label function sigma
	float scale_learning_rate = 0.025;		 // Scale filter learning rate
	int number_of_scales_filter = 17;		 // Number of scales
	int number_of_interp_scales = 33;		 // Number of interpolated scales
	float scale_model_factor = 1.0;			 // Scaling of the scale model
	float scale_step_filter = 1.02;			 // The scale factor for the scale filter
	float scale_model_max_area = 32 * 16;	 // Maximume area for the scale sample patch
	std::string scale_feature = "HOG4";	     // Features for the scale filter (only HOG4 supported)
	int s_num_compressed_dim = 17;	         // Number of compressed feature dimensions in the scale filter
	float lambda = 1e-2;					 // Scale filter regularization
	float do_poly_interp = true;			 // Do 2nd order polynomial interpolation to obtain more accurate scale
	cv::Size scale_model_sz;
};

class ScaleFilter {
public:
    ScaleFilter(){};
    virtual ~ScaleFilter(){};
    
public:
    void init(int &nScales, float &scale_step, const ScaleFilterParameter &params);
    float scale_filter_track(const cv::Mat &im, const cv::Point2f &pos, const cv::Size2f &base_target_sz,
        const float &currentScaleFactor, const ScaleFilterParameter &params);
    cv::Mat extract_scale_sample(const cv::Mat &im, const cv::Point2f &posf, const cv::Size2f &base_target_sz,
        std::vector<float> &scaleFactors, const cv::Size &scale_model_sz);
    
private:
    std::vector<float> scaleSizeFactors_;
    std::vector<float> interpScaleFactors_;
    cv::Mat yf_;
    std::vector<float> window_;
};

} // namespace cftracker

#endif

