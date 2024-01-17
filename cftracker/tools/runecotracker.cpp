#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "trackers/eco/eco_tracker.hpp"
#include "common/metrics.hpp"


using namespace cftracker;

int main(int argc, char **argv)
{
    std::vector<float> CenterError;
    std::vector<float> Iou;
    std::vector<float> FpsEco;
    float SuccessRate = 0.0f;
    float AvgPrecision = 0.0f;
    float AvgIou = 0.0f;
    float AvgFps = 0.0f;
    Metrics metrics;

    int f, isLost;
    float x, y, w, h;
    float x1, y1, x2, y2, x3, y3, x4, y4; //gt for vot
    std::string s;
    std::string path;
    std::ostringstream osfile;
    path = "../sequences/Crossing";
    // Read images in a folder
    osfile << path << "/img/" << std::setw(4) << std::setfill('0') << f << ".jpg";
    std::cout << osfile.str() << std::endl;

    cv::Rect2f bboxGroundtruth(x, y, w, h);

    cv::Mat frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat frameDraw;
    frame.copyTo(frameDraw);
    if (!frame.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    // Draw gt;
    rectangle(frameDraw, bboxGroundtruth, cv::Scalar(0, 0, 0), 2, 1);

    double timereco = (double)cv::getTickCount();
    EcoTracker ecotracker;
    cv::Rect2f ecobbox(x, y, w, h);
    EcoParameters parameters;

    ecotracker.init(frame, ecobbox, parameters);
    float fpsecoini = cv::getTickFrequency() / ((double)cv::getTickCount() - timereco);

    while (frame.data)
    {
        frame.copyTo(frameDraw); // only copy can do the real copy, just equal not.
        timereco = (double)cv::getTickCount();
        bool okeco = ecotracker.update(frame, ecobbox);
        float fpseco = cv::getTickFrequency() / ((double)cv::getTickCount() - timereco);
        if (okeco)
        {
            rectangle(frameDraw, ecobbox, cv::Scalar(255, 0, 255), 2, 1); //blue
        }
        else
        {
            putText(frameDraw, "ECO tracking failure detected", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 255), 2);
            //waitKey(0);
        }

        // Display FPS on frameDraw
        std::ostringstream os; 
        os << float(fpseco); 
        putText(frameDraw, "FPS: " + os.str(), cv::Point(100, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.75, cv::Scalar(255, 0, 255), 2);

        if (parameters.debug == 0)
        {
            imshow("OpenTracker", frameDraw);
        }

        int c = cvWaitKey(1);
        if (c != -1)
            c = c % 256;
        if (c == 27)
        {
            cvDestroyWindow("OpenTracker");
            exit(1);
        }
        cv::waitKey(1);
        // Read next image======================================================
        std::cout << "Frame:" << f << " FPS:" << fpseco << std::endl;
        f++;

        bboxGroundtruth.x = x;
        bboxGroundtruth.y = y;
        bboxGroundtruth.width = w;
        bboxGroundtruth.height = h;
        frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if(!frame.data)
        {
            break;
        }
        // Calculate the metrics;
        float centererror = metrics.center_error(ecobbox, bboxGroundtruth);
        float iou = metrics.iou(ecobbox, bboxGroundtruth);
        CenterError.push_back(centererror);
        Iou.push_back(iou);
        FpsEco.push_back(fpseco);

        std::cout << "iou:" << iou << std::endl;

        if(centererror <= 20)
        {
            AvgPrecision++;
        }
        if(iou >= 0.5)
        {
            SuccessRate++;
        }
    }
#ifdef USE_MULTI_THREAD
    void *status;
    if (pthread_join(ecotracker.thread_train_, &status))
    {
         cout << "Error:unable to join!"  << std::endl;
         exit(-1);
    }
#endif
    AvgPrecision /= (float)(f - 2);
    SuccessRate /= (float)(f - 2);
    AvgIou = std::accumulate(Iou.begin(), Iou.end(), 0.0f) / Iou.size();
    AvgFps = std::accumulate(FpsEco.begin(), FpsEco.end(), 0.0f) / FpsEco.size();
    std::cout << "Frames:" << f - 2
         << " AvgPrecision:" << AvgPrecision
         << " AvgIou:" << AvgIou 
         << " SuccessRate:" << SuccessRate
         << " IniFps:" << fpsecoini
         << " AvgFps:" << AvgFps << std::endl;

    return 0;
}

