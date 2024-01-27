#ifndef CFTRACKER_SAMPLE_UPDATE_HPP
#define CFTRACKER_SAMPLE_UPDATE_HPP

#include "cftracker/common/datatype.hpp"
#include "cftracker/feature/feature_operator.hpp"


namespace cftracker {

class SampleUpdate {
  public:
    SampleUpdate(){};
    virtual ~SampleUpdate(){};

  public:
    void init(const std::vector<cv::Size> &filter, const std::vector<int> &feature_dim,
              const size_t nSamples, const float learning_rate);
    void update_sample_space_model(const FEAT_DATA &new_train_sample); 

    inline void replace_sample(const FEAT_DATA &new_sample, const size_t idx) {
        samples_f_[idx] = new_sample;
    };
    inline void set_gram_matrix(const int r, const int c, const float val) {
        gram_matrix_.at<float>(r, c) = val;
    };
    int get_merged_sample_id() const { return merged_sample_id_; }
    int get_new_sample_id() const { return new_sample_id_; }
    std::vector<float> get_prior_weights() const { return prior_weights_; }
    std::vector<FEAT_DATA> get_samples() const { return samples_f_; }
  
  protected:
    void update_distance_matrix(cv::Mat &gram_vector, float new_sample_norm,
                                int id1, int id2, float w1, float w2);
    cv::Mat find_gram_vector(const FEAT_DATA &new_train_sample);
    FEAT_DATA merge_samples(const FEAT_DATA &sample1, const FEAT_DATA &sample2,
        const float w1, const float w2, const std::string sample_merge_type="merge");

  private:
    cv::Mat distance_matrix_, gram_matrix_; // distance matrix and its kernel
    size_t nSamples_ = 50;
    float learning_rate_ = 0.009;
    const float minmum_sample_weight_ = 0.0036;
    std::vector<float> sample_weight_;
    std::vector<FEAT_DATA> samples_f_; // all samples frontier
    size_t num_training_samples_ = 0;
    std::vector<float> prior_weights_;
    FEAT_DATA new_sample_, merged_sample_;
    int new_sample_id_ = -1, merged_sample_id_ = -1;
};

} // namespace cftracker

#endif // CFTRACKER_SAMPLE_UPDATE_HPP

