#include "tpMorphology.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <limits>
#include "common.h"
using namespace cv;
using namespace std;

namespace HelperFunctions
{
    float getGrayPixelSafe(const cv::Mat &img, int y, int x)
    {
        if (img.empty())
            return 0.0f;

        // should return 0 if out of bounds
        if (y < 0 || y >= img.rows || x < 0 || x >= img.cols)
            return 0.0f;

        return img.at<float>(y, x);
    }

    Mat inverse(Mat image) // taken from TP1
    {
        // clone original image
        Mat res = image.clone();
        /********************************************
                  YOUR CODE HERE
        *********************************************/

        for (int y = 0; y < res.rows; y++)
        {
            float *row = res.ptr<float>(y);
            for (int x = 0; x < res.cols; x++)
            {
                row[x] = 1.0 - image.at<float>(y, x);
            }
        }

        /********************************************
                    END OF YOUR CODE
        *********************************************/
        return res;
    }

}

/**
    Compute a median filter of the input float image.
    The filter window is a square of (2*size+1)*(2*size+1) pixels.

    Values outside image domain are ignored.

    The median of a list l of n>2 elements is defined as:
     - l[n/2] if n is odd
     - (l[n/2-1]+l[n/2])/2 is n is even
*/
Mat median(Mat image, int size)
{
    Mat res = image.clone();
    assert(size > 0);
    /********************************************
                YOUR CODE HERE
    *********************************************/

    int kernel_size = 2 * size + 1;

    vector<float> kernel;
    kernel.reserve(kernel_size * kernel_size);

    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);

        for (int x = 0; x < res.cols; x++)
        {
            kernel.clear();

            for (int ky = -size; ky <= size; ky++)
            {
                int neighbor_y = y + ky;

                if (neighbor_y < 0 || neighbor_y >= image.rows)
                {
                    continue;
                }

                for (int kx = -size; kx <= size; kx++)
                {
                    int neighbor_x = x + kx;

                    if (neighbor_x < 0 || neighbor_x >= image.cols)
                    {
                        continue;
                    }

                    kernel.push_back(image.at<float>(neighbor_y, neighbor_x));
                }
            }

            if (kernel.empty())
            {
                row[x] = image.at<float>(y, x); // fallback
                continue;
            }

            sort(kernel.begin(), kernel.end());

            int n = (int)kernel.size();

            if ((n % 2) == 1)
            {
                row[x] = kernel[n / 2]; // odd: middle element
            }
            else
            {
                row[x] = 0.5f * (kernel[n / 2 - 1] + kernel[n / 2]); // even: mean of two middle elements
            }
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the dilation of the input float image by the given structuring element.
     Pixel outside the image are supposed to have value 0
*/
Mat dilate(Mat image, Mat structuringElement)
{
    Mat res = Mat::zeros(image.size(), CV_32FC1);
    /********************************************
                YOUR CODE HERE
    *********************************************/

    int se_height = structuringElement.rows;
    int se_width = structuringElement.cols;

    int anchor_y = se_height / 2;
    int anchor_x = se_width / 2;

    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);

        for (int x = 0; x < res.cols; x++)
        {
            float max_value = 0.0f;

            for (int se_y = 0; se_y < se_height; se_y++)
            {
                for (int se_x = 0; se_x < se_width; se_x++)
                {
                    // skip SE zeros
                    float se_value = (structuringElement.at<float>(se_y, se_x) != 0) ? 1.0f : 0.0f;

                    if (se_value == 0.0f)
                    {
                        continue;
                    }

                    int neighbor_y = y + (se_y - anchor_y);
                    int neighbor_x = x + (se_x - anchor_x);

                    float neighbor_value = HelperFunctions::getGrayPixelSafe(image, neighbor_y, neighbor_x);

                    if (neighbor_value > max_value)
                    {
                        max_value = neighbor_value;
                    }
                }
            }

            row[x] = max_value;
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the erosion of the input float image by the given structuring element.
    Pixel outside the image are supposed to have value 1.
*/
Mat erode(Mat image, Mat structuringElement)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/

    res = HelperFunctions::inverse(dilate(HelperFunctions::inverse(image), structuringElement));

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the opening of the input float image by the given structuring element.
*/
Mat open(Mat image, Mat structuringElement)
{

    Mat res = Mat::zeros(1, 1, CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/

    res = dilate(erode(image, structuringElement), structuringElement);

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the closing of the input float image by the given structuring element.
*/
Mat close(Mat image, Mat structuringElement)
{

    Mat res = Mat::zeros(1, 1, CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/

    res = erode(dilate(image, structuringElement), structuringElement);

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the morphological gradient of the input float image by the given structuring element.
*/
Mat morphologicalGradient(Mat image, Mat structuringElement)
{

    Mat res = Mat::zeros(1, 1, CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/

    res = dilate(image, structuringElement) - erode(image, structuringElement);

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
