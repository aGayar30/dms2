#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "threadsafequeue.h"
#include <thread>
#include <chrono>


class HeadPoseComponent {
public:
    HeadPoseComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<std::string>& outputQueue,ThreadSafeQueue<std::string>& commandsQueue,ThreadSafeQueue<std::string>& faultsQueue);
    ~HeadPoseComponent();

    bool initialize();
    void startHeadPoseDetection();
    void stopHeadPoseDetection();



private:
    ThreadSafeQueue<cv::Mat>& inputQueue;
    ThreadSafeQueue<std::string>& outputQueue;
    ThreadSafeQueue<std::string>& commandsQueue;
    ThreadSafeQueue<std::string>& faultsQueue;
    std::thread HeadPoseDetectionThread;
    double avgDetectionTime;  // Average time for detection per frame

    bool running;
    
    void HeadPoseDetectionLoop();
    void detectHeadPose(cv::Mat& frame);

    
    // members for performance metrics
    double totalDetectionTime = 0;
    int totalFramesProcessed = 0;
    std::chrono::high_resolution_clock::time_point lastTime;
    double fps = 0;
    void updatePerformanceMetrics(double detectionTime);
    void displayPerformanceMetrics(cv::Mat& frame);
};

