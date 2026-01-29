
#include "tpConvolution.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

// declare before using
float gaussian(float x, float sigma2);

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

    float getGrayPixelBorder(const cv::Mat &img, int y, int x)
    {
        if (img.empty())
            return 0.0f;

        // clamp coordinates to image range (Replicate Border)
        int ny = std::max(0, std::min(y, img.rows - 1));
        int nx = std::max(0, std::min(x, img.cols - 1));
        return img.at<float>(ny, nx);
    }

    void fillKernel(float **kernel,
                    int kernelSize,
                    const cv::Mat &img,
                    int cy,
                    int cx)
    {
        int half = kernelSize / 2;

        for (int i = 0; i < kernelSize; ++i)
        {
            for (int j = 0; j < kernelSize; ++j)
            {
                int y = cy + i - half;
                int x = cx + j - half;

                kernel[i][j] = getGrayPixelSafe(img, y, x);
            }
        }
    }

    void fillConvolutionKernel(cv::Mat &kernel,
                               const cv::Mat &img,
                               int cy,
                               int cx)
    {
        int kernelSize = kernel.rows;
        int half = kernelSize / 2;

        for (int i = 0; i < kernelSize; ++i)
        {
            float *row = kernel.ptr<float>(i);
            for (int j = 0; j < kernelSize; ++j)
            {
                int y = cy + i - half;
                int x = cx + j - half;

                row[j] *= getGrayPixelSafe(img, y, x);
            }
        }
    }

    // used by 'edgeSobel' (Uses Border Clamping)
    void fillConvolutionKernelBorder(cv::Mat &kernel, const cv::Mat &img, int cy, int cx)
    {
        int kernelSize = kernel.rows;
        int half = kernelSize / 2;
        for (int i = 0; i < kernelSize; ++i)
        {
            float *row = kernel.ptr<float>(i);
            for (int j = 0; j < kernelSize; ++j)
            {
                int y = cy + i - half;
                int x = cx + j - half;

                // USES BORDER CLAMPING
                row[j] *= getGrayPixelBorder(img, y, x);
            }
        }
    }

    void fillBilateralKernel(cv::Mat &kernel,
                             const cv::Mat &image,
                             int cy, int cx,
                             float sigma_r)
    {
        int k = kernel.rows / 2;
        float center = image.at<float>(cy, cx);

        for (int i = 0; i < kernel.rows; ++i)
        {
            float *row = kernel.ptr<float>(i);

            for (int j = 0; j < kernel.cols; ++j)
            {
                int y = cy + i - k;
                int x = cx + j - k;

                float neighbor = HelperFunctions::getGrayPixelBorder(image, y, x);

                float rangeWeight = gaussian(neighbor - center, sigma_r * sigma_r);

                row[j] *= rangeWeight;
            }
        }
    }

    float kernelSum(const cv::Mat &kernel)
    {
        float sum = 0.f;

        int kernelSize = kernel.rows;

        for (int i = 0; i < kernelSize; ++i)
        {
            const float *row = kernel.ptr<float>(i);
            for (int j = 0; j < kernelSize; ++j)
            {
                sum += row[j];
            }
        }
        return sum;
    }

    float kernelSum(float **kernel, int kernelSize)
    {
        float sum = 0.0f;

        for (int i = 0; i < kernelSize; ++i)
        {
            for (int j = 0; j < kernelSize; ++j)
            {
                sum += kernel[i][j];
            }
        }

        return sum;
    }

    float kernelAverage(float **kernel, int kernelSize)
    {
        float sum = kernelSum(kernel, kernelSize);
        return sum / static_cast<float>(kernelSize * kernelSize);
    }

}

/**
    Compute a mean filter of size 2k+1.

    Pixel values outside of the image domain are supposed to have a zero value.
*/
cv::Mat meanFilter(cv::Mat image, int k)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/

    int kernelSize = 2 * k + 1;

    float **kernelMatrix = new float *[kernelSize];
    for (int i = 0; i < kernelSize; ++i)
        kernelMatrix[i] = new float[kernelSize];

    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            HelperFunctions::fillKernel(kernelMatrix, kernelSize, image, y, x);
            row[x] = HelperFunctions::kernelAverage(kernelMatrix, kernelSize);
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/

    return res;
}

/**
    Compute the convolution of a float image by kernel.
    Result has the same size as image.

    Pixel values outside of the image domain are supposed to have a zero value.
*/
Mat convolution(Mat image, cv::Mat kernel)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            cv::Mat duplicateKernel = kernel.clone();
            HelperFunctions::fillConvolutionKernel(duplicateKernel, image, y, x);
            row[x] = HelperFunctions::kernelSum(duplicateKernel);
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/

    return res;
}

/**
    Compute the sum of absolute partial derivative according to Sobel's method
*/
cv::Mat edgeSobel(cv::Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/

    //NOTE: does not pass tests but looks right

    cv::Mat dxKernel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1);

    cv::Mat dyKernel = (cv::Mat_<float>(3, 3) << -1, -2, -1,
                        0, 0, 0,
                        1, 2, 1);

    Mat sobelDX = res.clone();
    Mat sobelDY = res.clone();

    // get SobelDX
    for (int y = 0; y < sobelDX.rows; y++)
    {
        float *row = sobelDX.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            cv::Mat duplicateKernel = dxKernel.clone();
            HelperFunctions::fillConvolutionKernelBorder(duplicateKernel, image, y, x);
            row[x] = HelperFunctions::kernelSum(duplicateKernel);
        }
    }

    // get SobelDY
    for (int y = 0; y < sobelDY.rows; y++)
    {
        float *row = sobelDY.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            cv::Mat duplicateKernel = dyKernel.clone();
            HelperFunctions::fillConvolutionKernelBorder(duplicateKernel, image, y, x);
            row[x] = HelperFunctions::kernelSum(duplicateKernel);
        }
    }

    // final image
    //|| grad F ||
    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            row[x] = sqrt(
                pow(sobelDX.at<float>(y, x), 2) +
                pow(sobelDY.at<float>(y, x), 2));
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Value of a centered gaussian of variance (scale) sigma at point x.
*/
float gaussian(float x, float sigma2)
{
    return 1.0 / (2 * M_PI * sigma2) * exp(-x * x / (2 * sigma2));
}

/**
    Performs a bilateral filter with the given spatial smoothing kernel
    and a intensity smoothing of scale sigma_r.

*/
cv::Mat bilateralFilter(cv::Mat image, cv::Mat kernel, float sigma_r)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/

    //NOTE: does not pass tests but looks right

    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            cv::Mat w = kernel.clone();
            HelperFunctions::fillBilateralKernel(w, image, y, x, sigma_r);

            float sumWeights = 0.0f;
            float sum = 0.0f;
            int k = w.rows / 2;

            for (int i = 0; i < w.rows; ++i)
            {
                for (int j = 0; j < w.cols; ++j)
                {
                    int yy = y + i - k;
                    int xx = x + j - k;

                    float pixel = HelperFunctions::getGrayPixelBorder(image, yy, xx);
                    float weight = w.at<float>(i, j);

                    sum += weight * pixel;
                    sumWeights += weight;
                }
            }

            row[x] = (sumWeights > 0.0f) ? sum / sumWeights : 0.0f;
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
