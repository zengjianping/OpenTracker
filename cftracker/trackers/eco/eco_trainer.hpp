#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <iostream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "eco_parameter.hpp"
#include "feature/feature_data.hpp"


namespace cftracker {

class EcoTrainer {
public:
    EcoTrainer();
    virtual ~EcoTrainer();

    struct STATE {
        FEAT_DATA p, r_prev;
        float rho;
    };

    // the right and left side of the equation (18) of suppl. paper ECO
    struct ECO_EQ {
        ECO_EQ() {}
        ECO_EQ(FEAT_DATA up_part, std::vector<cv::Mat> low_part) : up_part_(up_part), low_part_(low_part) {}

        FEAT_DATA up_part_;             // this is f + delta(f)
        std::vector<cv::Mat> low_part_; // this is delta(P)

        ECO_EQ operator+(const ECO_EQ data);
        ECO_EQ operator-(const ECO_EQ data);
        ECO_EQ operator*(const float scale);
    };

    void train_init(const FEAT_DATA &hf,
                    const FEAT_DATA &hf_inc,
                    const std::vector<cv::Mat> &proj_matrix,
                    const FEAT_DATA &xlf,
                    const std::vector<cv::Mat> &yf,
                    const std::vector<cv::Mat> &reg_filter,
                    const FEAT_DATA &sample_energy,
                    const std::vector<float> &reg_energy,
                    const std::vector<cv::Mat> &proj_energy,
                    const EcoParameters &params);

    // Filter training and Projection updating(for the 1st Frame)
    void train_joint();

    ECO_EQ pcg_eco_joint(const FEAT_DATA &init_samplef_proj,
                         const std::vector<cv::Mat> &reg_filter,
                         const FEAT_DATA &init_samplef,
                         const std::vector<cv::Mat> &init_samplesf_H,
                         const FEAT_DATA &init_hf,
                         const ECO_EQ &rhs_samplef,
                         const ECO_EQ &diag_M, // preconditionor
                         const ECO_EQ &hf);

    ECO_EQ lhs_operation_joint(const ECO_EQ &hf,
                               const FEAT_DATA &samplesf,
                               const std::vector<cv::Mat> &reg_filter,
                               const FEAT_DATA &init_samplef,
                               const std::vector<cv::Mat> &XH,
                               const FEAT_DATA &init_hf);

    // Only filter training(for tracker update)
    void train_filter(const std::vector<FEAT_DATA> &samplesf,
                      const std::vector<float> &sample_weights,
                      const FEAT_DATA &sample_energy);

    FEAT_DATA pcg_eco_filter(const std::vector<FEAT_DATA> &samplesf,
                             const std::vector<cv::Mat> &reg_filter,
                             const std::vector<float> &sample_weights,
                             const FEAT_DATA &rhs_samplef,
                             const FEAT_DATA &diag_M,
                             const FEAT_DATA &hf);

    FEAT_DATA lhs_operation_filter(const FEAT_DATA &hf,
                                   const std::vector<FEAT_DATA> &samplesf,
                                   const std::vector<cv::Mat> &reg_filter,
                                   const std::vector<float> &sample_weights);
                                   
    // joint structure basic operation
    ECO_EQ jointDotDivision(const ECO_EQ &a, const ECO_EQ &b);
    float inner_product_joint(const ECO_EQ &a, const ECO_EQ &b);
    float inner_product_filter(const FEAT_DATA &a, const FEAT_DATA &b);
    std::vector<cv::Mat> get_proj() const { return projection_matrix_; }
    FEAT_DATA get_hf() const { return hf_; }

private:
    FEAT_DATA hf_, hf_inc_; // filter parameters and its increament
    FEAT_DATA xlf_, sample_energy_;
    std::vector<cv::Mat> yf_; // the label of sample
    std::vector<cv::Mat> reg_filter_;
    std::vector<float> reg_energy_;
    std::vector<cv::Mat> projection_matrix_, proj_energy_;
    EcoParameters params_;
    STATE state_;
};

} // namespace cftracker

#endif

