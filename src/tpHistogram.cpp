#include "tpHistogram.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

/**
    Inverse a grayscale image with float values.
    for all pixel p: res(p) = 1.0 - image(p)
*/
Mat inverse(Mat image)
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

/**
    Thresholds a grayscale image with float values.
    for all pixel p: res(p) =
        | 0 if image(p) <= lowT
        | image(p) if lowT < image(p) <= highT
        | 1 otherwise
*/
Mat threshold(Mat image, float lowT, float highT)
{
    Mat res = image.clone();
    assert(lowT <= highT);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    for (int y = 0; y < res.rows; y++)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; x++)
        {
            if (row[x] <= lowT)
            {
                row[x] = 0.0;
            }
            else if (row[x] > highT)
            {
                row[x] = 1.0;
            }
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Quantize the input float image in [0,1] in numberOfLevels different gray levels.

    eg. for numberOfLevels = 3 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/3
        | 1/2 if 1/3 <= image(p) < 2/3
        | 1 otherwise

        for numberOfLevels = 4 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/4
        | 1/3 if 1/4 <= image(p) < 1/2
        | 2/3 if 1/2 <= image(p) < 3/4
        | 1 otherwise

        and so on for other values of numberOfLevels.

*/
Mat quantize(Mat image, int numberOfLevels)
{
    Mat res = image.clone();
    assert(numberOfLevels > 0);
    /********************************************
                YOUR CODE HERE
    *********************************************/

    for (int y = 0; y < res.rows; ++y)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; ++x)
        {
            int i = static_cast<int>(row[x] * numberOfLevels);
            row[x] = i / float(numberOfLevels - 1);
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Normalize a grayscale image with float values
    Target range is [minValue, maxValue].
*/
Mat normalize(Mat image, float minValue, float maxValue)
{
    Mat res = image.clone();
    assert(minValue <= maxValue);
    /********************************************
                YOUR CODE HERE
    *********************************************/

    double minVal, maxVal;
    cv::minMaxLoc(res, &minVal, &maxVal);

    for (int y = 0; y < res.rows; ++y)
    {
        float *row = res.ptr<float>(y);
        for (int x = 0; x < res.cols; ++x)
        {
            row[x] = (row[x] - minVal) * (1.f / (maxVal - minVal));
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Equalize image histogram with unsigned char values ([0;255])

    Warning: this time, image values are unsigned chars but calculation will be done in float or double format.
    The final result must be rounded toward the nearest integer
*/
Mat equalize(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
   
    const int histSize = 256;
    vector<int> hist(histSize, 0);

    // histogram
    for (int y = 0; y < image.rows; ++y)
    {
        const uchar *row = image.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            hist[row[x]]++;
        }
    }

    // cumulative histogram (CDF)
    vector<double> cdf(histSize, 0.0);
    cdf[0] = hist[0];
    for (int i = 1; i < histSize; ++i)
    {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    double totalPixels = image.rows * image.cols;

    // mapping
    for (int y = 0; y < res.rows; ++y)
    {
        uchar *row = res.ptr<uchar>(y);
        for (int x = 0; x < res.cols; ++x)
        {
            int val = row[x];
            row[x] = static_cast<uchar>(std::round(cdf[val] / totalPixels * 255.0));
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute a binarization of the input image using an automatic Otsu threshold.
    Input image is of type unsigned char ([0;255])
*/
Mat thresholdOtsu(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/

    const int histSize = 256;
    vector<int> hist(histSize, 0);

    // histogram
    for (int y = 0; y < image.rows; ++y)
    {
        const uchar *row = image.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            hist[row[x]]++;
        }
    }

    int total = image.rows * image.cols;

    // global mean
    double sum = 0.0;
    for (int i = 0; i < histSize; ++i)
    {
        sum += i * hist[i];
    }

    double sumB = 0.0;
    int wB = 0;
    int wF = 0;

    double maxVar = 0.0;
    int bestT = 0;

    for (int t = 0; t < histSize; ++t)
    {
        wB += hist[t];
        if (wB == 0)
            continue;

        wF = total - wB;
        if (wF == 0)
            break;

        sumB += t * hist[t];

        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);

        if (varBetween > maxVar)
        {
            maxVar = varBetween;
            bestT = t;
        }
    }

    // apply threshold
    for (int y = 0; y < res.rows; ++y)
    {
        uchar *row = res.ptr<uchar>(y);
        for (int x = 0; x < res.cols; ++x)
        {
            if (row[x] <= bestT)
                row[x] = 0;
            else
                row[x] = 255;
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
