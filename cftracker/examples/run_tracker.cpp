#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include "cftracker/trackers/base_tracker.hpp"
#include "cftracker/common/metrics.hpp"

using namespace cftracker;


void find_file_on_pattern(const std::string file_pattern, std::vector<std::string>& file_paths) {
    boost::filesystem::path fs_path(file_pattern);
    std::string pattern = fs_path.filename().string();
    const boost::regex regex_filter(pattern);
    fs_path = fs_path.parent_path();

    for (auto i = boost::filesystem::directory_iterator(fs_path);
            i != boost::filesystem::directory_iterator(); i++) {
        if (boost::filesystem::is_regular_file(i->status())) {
            std::string filename = i->path().filename().string();
            if (boost::regex_match(filename, regex_filter)) {
                //std::cout << filename << std::endl;
                file_paths.push_back(i->path().string());
            }
        }
    }
    std::sort(file_paths.begin(), file_paths.end());
}

class VideoReader {
public:
    VideoReader() = delete;

    VideoReader(const char* video_path) {
        boost::filesystem::path fs_path(video_path);
        boost::filesystem::file_status fs_status = boost::filesystem::status(fs_path);

        if (boost::filesystem::is_regular_file(fs_status)) {
            video_capture_ = cv::VideoCapture(video_path);
            if (video_capture_.isOpened()) {
                input_mode_ = 0;
            }
        }
        else if (boost::filesystem::is_directory(fs_status)) {
            boost::filesystem::path image_path(video_path);
            image_path.append("img").append(".*\\.jpg");
            std::string image_pattern = image_path.string();
            find_file_on_pattern(image_pattern, image_file_paths_);

            if (image_file_paths_.size() > 0) {
                std::string gt_file = boost::filesystem::path(video_path).append("groundtruth_rect.txt").string();
                std::ifstream gt_stream(gt_file);
                while (!gt_stream.eof()) {
                    float x, y, w, h;
                    std::string s;
                    std::getline(gt_stream, s, ',');  x = atof(s.c_str());
                    std::getline(gt_stream, s, ',');  y = atof(s.c_str());
                    std::getline(gt_stream, s, ',');  w = atof(s.c_str());
                    std::getline(gt_stream, s, '\n'); h = atof(s.c_str());
                    gt_bboxs_.push_back(cv::Rect2f(x,y,w,h));
                    //std::cout << cv::Rect2f(x,y,w,h) << std::endl;
                }
                std::cout << "Number of image files: " << image_file_paths_.size() << std::endl;
                std::cout << "Number of gt bboxs: " << gt_bboxs_.size() << std::endl;

                frame_index_ = 0;
                input_mode_ = 1;
            }
        }
    }

    bool IsOpened() {
        return input_mode_ >= 0;
    }

    bool HasGroundTruth() {
        return gt_bboxs_.size() > 0;
    }

    bool Retrieve(cv::Mat& frame, cv::Rect2f* gt_bbox) {
        if (input_mode_ == 0) {
            if (video_capture_.grab()) {
                return video_capture_.retrieve(frame);
            }
        }
        else if(input_mode_ == 1) {
            if (frame_index_ < (int)image_file_paths_.size()) {
                if (gt_bboxs_.size() > 0 && gt_bbox) {
                    *gt_bbox = gt_bboxs_[frame_index_];
                }
                const std::string& file_path = image_file_paths_[frame_index_];
                frame = cv::imread(file_path, cv::IMREAD_COLOR);
                frame_index_++;
                return frame.cols > 0;
            }
        }
        return false;
    }

    double GetFps() {
        if (input_mode_ == 0) {
            return video_capture_.get(CV_CAP_PROP_FPS);
        }
        else if (input_mode_ == 1) {
            return 1.0;
        }
        return 0;
    }

    double GetProgress() {
        if (input_mode_ == 0) {
            return video_capture_.get(CV_CAP_PROP_POS_MSEC);
        }
        else if (input_mode_ == 1) {
            return 1.0;
        }
        return 0;
    }

    void SetProgress(double time_ms) {
        if (input_mode_ == 0) {
            //video_capture_.set(cv::CAP_PROP_POS_MSEC, time_ms);
            video_capture_.set(CV_CAP_PROP_POS_FRAMES, time_ms/1000*video_capture_.get(CV_CAP_PROP_FPS));
        }
        else if (input_mode_ == 1) {
        }
    }

    void Release() {
        if (input_mode_ == 0) {
            video_capture_.release();
        }
        else if (input_mode_ == 1) {
            frame_index_ = -1;
        }
        input_mode_ = -1;
    }

protected:
    int input_mode_ = -1;
    cv::VideoCapture video_capture_;
    std::vector<std::string> image_file_paths_;
    int frame_index_ = -1;
    std::vector<cv::Rect2f> gt_bboxs_;
};


DEFINE_string(video_path, "datas/sequences/Crossing", "video path.");
DEFINE_string(tracker_name, "ECO", "tracker algorithm type.");
DEFINE_string(tracker_config, "configs/tracker_config_eco.yaml", "configuration file.");
DEFINE_bool(select_roi, false, "Whether to select target region.");

std::map<std::string, BaseTracker::Type> tracker_type_map = {
    {"ECO", BaseTracker::ECO},
    {"KCF", BaseTracker::KCF},
    {"DSST", BaseTracker::DSST}
};

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    std::string video_path = FLAGS_video_path;
    std::string tracker_name = FLAGS_tracker_name;
    std::string tracker_config = FLAGS_tracker_config;
    bool select_roi = FLAGS_select_roi;

    // Read the groundtruth bbox
    VideoReader video_reader = VideoReader(video_path.c_str());
    if (!video_reader.IsOpened()) {
        std::cout << "Failed to open video!" << std::endl;
        return -1;
    }

    if (tracker_type_map.find(tracker_name) == tracker_type_map.end()) {
        std::cout << "Unsupport tracker type!" << std::endl;
        return -1;
    }
    BaseTracker::Type tracker_type = tracker_type_map[tracker_name];

    std::shared_ptr<BaseTracker> tracker = BaseTracker::CreateInstance(tracker_type, tracker_config);
    if (!tracker.get()) {
        std::cout << "Failed to create tracker!" << std::endl;
        return -1;
    }

    cv::Rect2f gt_bbox, *p_gt_bbox = nullptr;
    if (video_reader.HasGroundTruth() && !select_roi)
        p_gt_bbox = &gt_bbox;
    cv::Rect2f init_bbox, updated_bbox, tracked_bbox;
    cv::Mat frame, frame_draw;
    double time_begin, time_init, time_update, time_track;
    bool track_initialized = false;
    int num_frames = 0, num_updated_frames = 0;;

    std::vector<float> m_errors;
    std::vector<float> m_ious;
    std::vector<float> m_times;
    float success_ratio = 0.0f;
    float avg_precision = 0.0f;
    float avg_iou = 0.0f;
    float avg_time = 0.0f;
    Metrics metrics;

    while (true) {
        if (!video_reader.Retrieve(frame, p_gt_bbox)) {
            std::cout << "Video ended!" << std::endl;
            break;
        }
        num_frames++;
        frame.copyTo(frame_draw);

        bool suc_done = true;
        if (!track_initialized) {
            if (p_gt_bbox) {
                init_bbox = *p_gt_bbox;
            }
            else {
                init_bbox = cv::selectROI("init target", frame, false, false);
            }

            time_begin = (double)cv::getTickCount();
            tracker->init(frame, init_bbox);
            time_init = (cv::getTickCount() - time_begin) / cv::getTickFrequency() * 1000;

            time_track = time_init;
            tracked_bbox = init_bbox;
            track_initialized = true;
        }
        else {
            time_begin = (double)cv::getTickCount();
            suc_done = tracker->update(frame, updated_bbox);
            time_update = (cv::getTickCount() - time_begin) / cv::getTickFrequency() * 1000;

            time_track = time_update;
            tracked_bbox = updated_bbox;
            num_updated_frames++;

            // calculate the metrics
            if (p_gt_bbox) {
                float error = metrics.center_error(updated_bbox, gt_bbox);
                float iou = metrics.iou(updated_bbox, gt_bbox);
                m_errors.push_back(error);
                m_ious.push_back(iou);
                m_times.push_back(time_update);
                std::cout << "iou:" << iou << std::endl;
                if(error <= 20) {
                    avg_precision++;
                }
                if(iou >= 0.5) {
                    success_ratio++;
                }
            }
        }

        if (suc_done) {
            cv::rectangle(frame_draw, tracked_bbox, cv::Scalar(255, 0, 255), 2, 1);
        }
        else {
            cv::putText(frame_draw, "tracking failed", cv::Point(100, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 255), 2);
        }
        std::ostringstream os; 
        os << "FPS: " << 1000 / time_track;
        std::string fps_str = os.str();
        cv::putText(frame_draw, fps_str, cv::Point(100, 30), cv::FONT_HERSHEY_SIMPLEX,
            0.75, cv::Scalar(255, 0, 255), 2);
        std::cout << "Frame:" << num_frames << " " << fps_str << std::endl;
        cv::imshow("tracking target", frame_draw);

        int c = cv::waitKey(1);
        if (c == 27) {
            cv::destroyWindow("tracking target");
            return 0;
        }
    }

    tracker->release();

    if (p_gt_bbox && num_updated_frames > 0) {
        avg_precision /= num_updated_frames;
        success_ratio /= num_updated_frames;
        avg_iou = std::accumulate(m_ious.begin(), m_ious.end(), 0.0f) / m_ious.size();
        avg_time = std::accumulate(m_times.begin(), m_times.end(), 0.0f) / m_times.size();
        std::cout << "Frames:" << num_frames
            << " AvgPrecision:" << avg_precision
            << " SuccessRate:" << success_ratio
            << " AvgIou:" << avg_iou 
            << " InitTime:" << time_init
            << " AvgTime:" << avg_time << std::endl;
    }

    return 0;
}

