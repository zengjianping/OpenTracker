#ifndef CFTRACKER_FEATURE_EXTRACTOR_HPP
#define CFTRACKER_FEATURE_EXTRACTOR_HPP

#include "cftracker/feature/feature_data.hpp"
#include "cftracker/feature/feature_parameter.hpp"

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/caffe.hpp>
#endif


namespace cftracker {

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

#endif // CFTRACKER_FEATURE_EXTRACTOR_HPP

