#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <opencv2/core/core.hpp>
#include "feature_data.hpp"


namespace cftracker {

// cnn   feature   configuration
#ifdef USE_CAFFE
struct CnnParameters {
    string proto = "model/imagenet-vgg-m-2048.prototxt";
    string model = "model/VGG_CNN_M_2048.caffemodel";
    string mean_file = "model/VGG_mean.binaryproto";

    boost::shared_ptr<caffe::Net<float>> net;
    cv::Mat deep_mean_mat, deep_mean_mean_mat;

    string nn_name = "imagenet-vgg-m-2048.mat";
    std::vector<int> stride = {2, 16};            // stride in total
    std::vector<int> cell_size = {4, 16};        // downsample_factor
    std::vector<int> output_layer = {3, 14};        // Which layers to use
    std::vector<int> downsample_factor = {2, 1}; // How much to downsample each output layer
    int input_size_scale = 1;                // Extra scale factor of the input samples to the network (1 is no scaling)
    std::vector<int> nDim = {96, 512};            // Original dimension of features (ECO Paper Table 1)
    std::vector<int> compressed_dim = {16, 64};  // Compressed dimensionality of each output layer (ECO Paper Table 1)
    std::vector<float> penalty = {0, 0};

    std::vector<int> start_ind = {3, 3, 1, 1};     // sample feature start index
    std::vector<int> end_ind = {106, 106, 13, 13}; // sample feature end index
};

struct CnnFeatures
{
    CnnParameters fparams;
    cv::Size img_input_sz = cv::Size(224, 224); // VGG default input sample size
    cv::Size img_sample_sz;                        // the size of sample
    cv::Size data_sz_block0, data_sz_block1;
    cv::Mat mean;
};
#endif

// hog parameters cofiguration
struct HogParameters {
    int cell_size = 6;
    int compressed_dim = 10; // Compressed dimensionality of each output layer (ECO Paper Table 1)
    int nOrients = 9;
    size_t nDim = 31; // Original dimension of feature
    float penalty = 0;
};

struct HogFeatures {
    HogParameters fparams;
    cv::Size img_input_sz;  // input sample size
    cv::Size img_sample_sz; // the size of sample
    cv::Size data_sz_block0;
};

// CN parameters configuration
struct ColorspaceParameters {
    std::string colorspace = "gray";
    int cell_size = 1;
};

struct ColorspaceFeatures {
    ColorspaceParameters fparams;
    cv::Size img_input_sz;  
    cv::Size img_sample_sz;
    cv::Size data_sz_block0;
};

// only used for Color image
struct CnParameters {
    std::string tablename = "look_tables/CNnorm.txt";
    float table[32768][10];
    int cell_size = 4;
    int compressed_dim = 3;
    size_t nDim = 10; 
    float penalty = 0;
};

struct CnFeatures
{
    CnParameters fparams;
    cv::Size img_input_sz; 
    cv::Size img_sample_sz; 
    cv::Size data_sz_block0;
};

// only used for gray image
struct IcParameters {
    std::string tablename = "look_tables/intensityChannelNorm6";
    float table[256][5];
    int cell_size = 4;
    int compressed_dim = 3;
    size_t nDim = 5; 
    float penalty = 0;
};

struct IcFeatures
{
    IcParameters fparams;
    cv::Size img_input_sz;
    cv::Size img_sample_sz;
    cv::Size data_sz_block0;
};

// Parameters
struct FeatureParameters
{
	// Features
	bool useDeepFeature = false;
	bool useHogFeature = true;
	bool useColorspaceFeature = false;// not implemented yet
	bool useCnFeature = false;
	bool useIcFeature = true;

#ifdef USE_CAFFE
	CnnFeatures cnn_features;
#endif
	HogFeatures hog_features;
	ColorspaceFeatures colorspace_feature;
	CnFeatures cn_features;
	IcFeatures ic_features;
};

class FeatureExtractor
{
public:
    FeatureExtractor() {}
    virtual ~FeatureExtractor() {};

public:
    FEAT_DATA extractor(const cv::Mat image, const cv::Point2f pos, const std::vector<float> scales,
        const FeatureParameters &params, const bool &is_color_image);

public:
    std::vector<cv::Mat> get_hog_features_simd(const std::vector<cv::Mat> ims);
    std::vector<cv::Mat> get_hog_features(const std::vector<cv::Mat> ims);
    std::vector<cv::Mat> hog_feature_normalization(std::vector<cv::Mat> &hog_feat_maps);

    std::vector<cv::Mat> get_cn_features(const std::vector<cv::Mat> ims);
    std::vector<cv::Mat> cn_feature_normalization(std::vector<cv::Mat> &cn_feat_maps);

#ifdef USE_CAFFE
    FEAT_DATA get_cnn_layers(std::vector<cv::Mat> im, const cv::Mat &deep_mean_mat);
    cv::Mat sample_pool(const cv::Mat &im, int smaple_factor, int stride);
    void cnn_feature_normalization(FEAT_DATA &feature);
    inline FEAT_DATA get_cnn_feats() const { return cnn_feat_maps_; }
#endif

private:
    cv::Mat sample_patch(const cv::Mat im, const cv::Point2f pos, cv::Size2f sample_sz, cv::Size2f input_sz);

private:
    FeatureParameters params_;
    HogFeatures hog_features_;
    ColorspaceFeatures colorspace_features_;
    CnFeatures cn_features_;
    IcFeatures ic_features_;

#ifdef USE_CAFFE
    boost::shared_ptr<caffe::Net<float>> net_;
    CnnFeatures cnn_features_;
    int cnn_feat_ind_ = -1;
    FEAT_DATA cnn_feat_maps_;
#endif
};

} // namespace cftracker

#endif

