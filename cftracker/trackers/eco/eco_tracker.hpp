#ifndef ECO_TRACKER_HPP
#define ECO_TRACKER_HPP

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#endif

#ifdef USE_MULTI_THREAD
#include <pthread.h>
#include <unistd.h>
#endif

#include "eco_parameter.hpp"
#include "eco_trainer.hpp"
#include "sample_update.hpp"
#include "feature/feature_extractor.hpp"
#include "filters/scale_filter.hpp"


namespace cftracker
{
class EcoTracker
{
public:
    EcoTracker() {};
    virtual ~EcoTracker() {}

    void init(cv::Mat &im, const cv::Rect2f &rect, const EcoParameters &paramters); 
    bool update(const cv::Mat &frame, cv::Rect2f &roi);
  
    void init_parameters(const EcoParameters &parameters);
    void init_features(); 
#ifdef USE_CAFFE
    void read_deep_mean(const string &mean_file);
#endif
    void yf_gaussian(); // the desired outputs of features, real part of (9) in paper C-COT
    void cos_window(); // construct cosine window of features;
    FEAT_DATA interpolate_dft(const FEAT_DATA &xlf, std::vector<cv::Mat> &interp1_fs, std::vector<cv::Mat> &interp2_fs);
    FEAT_DATA compact_fourier_coeff(const FEAT_DATA &xf);
    FEAT_DATA full_fourier_coeff(const FEAT_DATA &xf);
    std::vector<cv::Mat> project_mat_energy(std::vector<cv::Mat> proj, std::vector<cv::Mat> yf);
    FEAT_DATA shift_sample(FEAT_DATA &xf, cv::Point2f shift, std::vector<cv::Mat> kx, std::vector<cv::Mat> ky);
#ifdef USE_MULTI_THREAD
    static void *thread_train(void *params);
#endif

private:
    bool is_color_image_;
    EcoParameters params_;
    cv::Point2f pos_; // final result
    size_t frames_since_last_train_; // used for update;

    // The max size of feature and its index, output_sz is T in (9) of C-COT paper
    size_t output_size_, output_index_;     

    cv::Size2f base_target_size_; // target size without scale
    cv::Size2i img_sample_size_; // base_target_sz * sarch_area_scale
    cv::Size2i img_support_size_; // the corresponding size in the image

    std::vector<cv::Size> feature_size_, filter_size_;
    std::vector<int> feature_dim_, compressed_dim_;

    ScaleFilter scale_filter_;
    int nScales_; // number of scales;
    float scale_step_;
    std::vector<float> scale_factors_;
    float currentScaleFactor_; // current img scale 

    // Compute the Fourier series indices 
    // kx_, ky_ is the k in (9) of C-COT paper, yf_ is the left part of (9);
    std::vector<cv::Mat> ky_, kx_, yf_; 
    std::vector<cv::Mat> interp1_fs_, interp2_fs_; 
    std::vector<cv::Mat> cos_window_;
    std::vector<cv::Mat> projection_matrix_;
    std::vector<cv::Mat> reg_filter_;
    std::vector<float> reg_energy_;
    FEAT_DATA sample_energy_;
    FEAT_DATA hf_full_;

    FeatureExtractor feature_extractor_;
    SampleUpdate sample_update_;
    EcoTrainer eco_trainer_;

#ifdef USE_MULTI_THREAD
    bool thread_flag_train_;
public:
    pthread_t thread_train_;
#endif
};

} // namespace cftracker

#endif // ECO_TRACKER_HPP


