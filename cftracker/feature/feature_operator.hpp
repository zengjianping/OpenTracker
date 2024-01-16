#ifndef FEATURE_OPERATOR_HPP
#define FEATURE_OPERATOR_HPP

#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include "feature_data.hpp"


namespace cftracker {

template <typename T>
extern std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> result;
    for (unsigned int i = 0; i < a.size(); ++i) {
        result.push_back(a[i] + b[i]);
    }
    return result;
}

template <typename T>
extern std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> result;
    for (unsigned int i = 0; i < a.size(); ++i) {
        result.push_back(a[i] - b[i]);
    }
    return result;
}

template <typename T>
extern std::vector<T> operator*(const std::vector<T> &a, const float scale) {
    std::vector<T> result;
    for (unsigned int i = 0; i < a.size(); ++i) {
        result.push_back(a[i] * scale);
    }
    return result;
}

extern FEAT_DATA do_dft(const FEAT_DATA &xlw);
extern FEAT_DATA do_windows(const FEAT_DATA &xl, std::vector<cv::Mat> &cos_win);

extern void FilterSymmetrize(FEAT_DATA &hf);
extern std::vector<cv::Mat> init_projection_matrix(const FEAT_DATA &init_sample,
        const std::vector<int> &compressed_dim, const std::vector<int> &feature_dim);
extern FEAT_DATA FeatureProjection(const FEAT_DATA &x,
                                   const std::vector<cv::Mat> &projection_matrix);
extern FEAT_DATA FeatureProjectionMultScale(const FEAT_DATA &x,
                                            const std::vector<cv::Mat> &projection_matrix);

extern float FeatureComputeInnerProduct(const FEAT_DATA &feat1, const FEAT_DATA &feat2);
extern float FeatureComputeEnergy(const FEAT_DATA &feat);
extern FEAT_DATA FeautreComputePower2(const FEAT_DATA &feats);
extern std::vector<cv::Mat> FeatureComputeScores(const FEAT_DATA &x, const FEAT_DATA &f);
extern std::vector<cv::Mat> FeatureVectorization(const FEAT_DATA &x);

extern FEAT_DATA FeatureVectorMultiply(const FEAT_DATA &x, const std::vector<cv::Mat> &y,
                                       const bool _conj = 0);

extern FEAT_DATA FeatureDotMultiply(const FEAT_DATA &a, const FEAT_DATA &b);
extern FEAT_DATA FeatureDotDivide(const FEAT_DATA &a, const FEAT_DATA &b);

} // namespace namespace cftracker {


#endif
