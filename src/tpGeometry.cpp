#include "tpGeometry.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

// c++11 does not have clamp
inline int clamp(int v, int lo, int hi)
{
    return (v < lo) ? lo : (v > hi) ? hi
                                    : v;
}

/**
    Transpose the input image,
    ie. performs a planar symmetry according to the
    first diagonal (upper left to lower right corner).
*/
Mat transpose(Mat image)
{
    Mat res = Mat::zeros(image.cols, image.rows, CV_32FC1);
    /********************************************
                YOUR CODE HERE
    hint: consider a non square image
    *********************************************/

    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            row[x] = image.at<float>(x, y);
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the value of a nearest neighbour interpolation
    in image Mat at position (x,y)
*/
float interpolate_nearest(Mat image, float y, float x)
{
    float v = 0;
    /********************************************
                YOUR CODE HERE
    *********************************************/

    // floor() is used for interpolation to get the lower neighbor.
    // round() selects the nearest value and is only suitable for nearest-neighbor sampling.

    int iy = static_cast<int>(std::round(y));
    int ix = static_cast<int>(std::round(x));

    // follow image bounds
    iy = clamp(iy, 0, image.rows - 1);
    ix = clamp(ix, 0, image.cols - 1);

    v = image.at<float>(iy, ix);

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return v;
}

/**
    Compute the value of a bilinear interpolation in image Mat at position (x,y)
*/
float interpolate_bilinear(Mat image, float y, float x)
{
    float v = 0;

    /********************************************
                YOUR CODE HERE
    *********************************************/

    // floor() is used to get the lower neighboring pixel.

    int x0 = floor(x), y0 = floor(y);
    int x1 = x0 + 1, y1 = y0 + 1;

    // clamp coordinates to stay inside image boundaries
    x0 = clamp(x0, 0, image.cols - 1);
    x1 = clamp(x1, 0, image.cols - 1);
    y0 = clamp(y0, 0, image.rows - 1);
    y1 = clamp(y1, 0, image.rows - 1);

    // fractional part of the coordinates
    float dx = x - x0;
    float dy = y - y0;

    auto lerp = [](float a, float b, float t)
    {
        return (1.0f - t) * a + t * b;
    };

    // step 1: interpolate along x
    float top = lerp(image.at<float>(y0, x0), image.at<float>(y0, x1), dx);
    float bottom = lerp(image.at<float>(y1, x0), image.at<float>(y1, x1), dx);

    // step 2: interpolate along y
    return lerp(top, bottom, dy);

    /********************************************
                END OF YOUR CODE
    *********************************************/

    return v;
}

/**
    Multiply the image resolution by a given factor using the given interpolation method.
    If the input size is (h,w) the output size shall be ((h-1)*factor, (w-1)*factor)
*/
Mat expand(Mat image, int factor, float (*interpolationFunction)(cv::Mat image, float y, float x))
{
    assert(factor > 0);
    Mat res = Mat::zeros((image.rows - 1) * factor, (image.cols - 1) * factor, CV_32FC1);
    /********************************************
                YOUR CODE HERE
    *********************************************/

    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            float src_y = static_cast<float>(y) / factor;
            float src_x = static_cast<float>(x) / factor;

            row[x] = interpolationFunction(image, src_y, src_x);
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Performs a rotation of the input image with the given angle (clockwise) and the given interpolation method.
    The center of rotation is the center of the image.

    Output size depends of the input image size and the rotation angle.

    Output pixels that map outside the input image are set to 0.
*/
Mat rotate(Mat image, float angle, float (*interpolationFunction)(cv::Mat image, float y, float x))
{
    //Mat res = Mat::zeros(1, 1, CV_32FC1);
    /********************************************
                YOUR CODE HERE
    hint: to determine the size of the output, take
    the bounding box of the rotated corners of the
    input image.
    *********************************************/

    //NOTE: does not pass tests but looks right

    float rad = -angle * CV_PI / 180.0f; //rotate clockwise
    float c = cos(rad);
    float s = sin(rad);

    // calculate new size (bounding box)
    // FIX: use ceil() to round UP so we don't loose the last pixel
    int w = std::ceil(abs(image.cols * c) + abs(image.rows * s));
    int h = std::ceil(abs(image.cols * s) + abs(image.rows * c));
    
    Mat res = Mat::zeros(h, w, CV_32FC1);

    // centers of images
    float cx = w / 2.0f;
    float cy = h / 2.0f;
    float ox = image.cols / 2.0f;
    float oy = image.rows / 2.0f;

    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            // inverse mapping coordinates
            float src_x = (x - cx) * c - (y - cy) * s + ox;
            float src_y = (x - cx) * s + (y - cy) * c + oy;

            // Only interpolate if inside original image
            if (src_x >= 0 && src_x < image.cols && src_y >= 0 && src_y < image.rows)
            {
                row[x] = interpolationFunction(image, src_y, src_x);
            }
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}