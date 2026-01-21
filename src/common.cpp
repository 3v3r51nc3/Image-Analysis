#include "common.h"
#include <exception>
#include <iostream>
#include <map>
#include "stdio.h"

using namespace cv;
using namespace std;

cv::Mat imreadHelper(std::string filename, bool forceFloat, bool forceGrayScale)
{
    cv::Mat image;
    image = cv::imread( filename.c_str(), (forceGrayScale)? 0 : -1 );

    if( !image.data )
    {
        throw std::runtime_error("No Image Data");
    }
    cv::Mat tmp;

    if(forceFloat)
    {
        int channels = image.channels();
        int depth = image.depth();
        int targetType;
        if(channels == 1)
            targetType = CV_32FC1;
        else if(channels == 3)
            targetType = CV_32FC3;
        else
            throw std::runtime_error("Unsupported number of channels");


        image.convertTo(tmp, targetType);

        if(depth<=1)
            tmp /= 255.0;
    } else {
        tmp = image;
    }


    return tmp;

}

void imwriteHelper(cv::Mat image, std::string filename)
{
    int depth = image.depth();
    if (depth == CV_8U) {
        cv::imwrite(filename.c_str(), image);
    } else {
        double min, max;
        cv::minMaxLoc(image, &min, &max);
        if (min < -0.000001 || max > 1.000001)
            std::cerr << "!!!  Warning, saved image values not between 0 and 1." << std::endl;

        cv::Mat tmp;
        // For float/double images in [0,1], scale to [0,255] and convert to 8U
        image.convertTo(tmp, CV_8U, 255.0);
        cv::imwrite(filename.c_str(), tmp);
    }

}

void showimage(cv::Mat image, const char * name)
{
    static int count = 1;
    static int current_x = 0;
    static int current_y = 0;
    static int max_row_height = 0;
    static const int screen_width = 1920;
    const char * tn;
    if (name == NULL)
    {
        char op[30];
        snprintf(op, 30, "%d", count);
        tn = op;
    }
    else
    {
        tn = name;
    }

    int win_width = image.cols;
    int win_height = image.rows;

    // If next window would go off screen, wrap to next row
    if (current_x + win_width > screen_width) {
        current_x = 0;
        current_y += max_row_height + 40; // 40px vertical gap
        max_row_height = 0;
    }

    cv::namedWindow(tn, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(tn, current_x, current_y);
    cv::imshow(tn, image);

    current_x += win_width + 40; // 40px horizontal gap
    if (win_height > max_row_height) max_row_height = win_height;
    count++;
}


cv::Mat remap_labels(cv::Mat label_image)
{
    map<int, int> label_map;
    label_map[0] = 0;

    Mat res = Mat::zeros(label_image.rows, label_image.cols, CV_32SC1);
    int current_label = 1;

    for(int y = 0; y < label_image.rows; ++y){
        for(int x = 0; x < label_image.cols; ++x){
            int l = label_image.at<int>(y, x);
            if(label_map.count(l) == 0){
                label_map[l] = current_label;
                current_label++;
            }
            res.at<int>(y, x) = label_map[l];
        }
    }

    return res;
}