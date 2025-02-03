/**
 * @file main.cpp
 * @brief Entry point for the Gesture Recognition Project
 * @author Daniel Gebura
 */

#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // Open The Camera
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    // Play video from camera
    cv::Mat frame;
    while (cap.read(frame)) {
        cap >> frame;
        cv::imshow("Camera", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the objects and destroy all windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}