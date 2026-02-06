#include "tpConnectedComponents.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <map>
#include <unordered_map>
#include <cassert>
using namespace cv;
using namespace std;

/**
    Utility: test if pixel is foreground (non-zero), for common single-channel types.
    This avoids assuming CV_8U in the tests.
*/
static inline bool isForeground(Mat image, int y, int x)
{
    switch (image.depth())
    {
    case CV_8U:
        return image.at<uchar>(y, x) != 0;
    case CV_8S:
        return image.at<schar>(y, x) != 0;
    case CV_16U:
        return image.at<ushort>(y, x) != 0;
    case CV_16S:
        return image.at<short>(y, x) != 0;
    case CV_32S:
        return image.at<int>(y, x) != 0;
    case CV_32F:
        return image.at<float>(y, x) != 0.0f;
    case CV_64F:
        return image.at<double>(y, x) != 0.0;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported image depth");
    }
    return false;
}

/**
    Utility: copy one pixel from src to dst for the same (single-channel) type.
*/
static inline void copyPixel(Mat src, Mat dst, int y, int x)
{
    switch (src.depth())
    {
    case CV_8U:
        dst.at<uchar>(y, x) = src.at<uchar>(y, x);
        break;
    case CV_8S:
        dst.at<schar>(y, x) = src.at<schar>(y, x);
        break;
    case CV_16U:
        dst.at<ushort>(y, x) = src.at<ushort>(y, x);
        break;
    case CV_16S:
        dst.at<short>(y, x) = src.at<short>(y, x);
        break;
    case CV_32S:
        dst.at<int>(y, x) = src.at<int>(y, x);
        break;
    case CV_32F:
        dst.at<float>(y, x) = src.at<float>(y, x);
        break;
    case CV_64F:
        dst.at<double>(y, x) = src.at<double>(y, x);
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported image depth");
    }
}

/**
    Performs a labeling of image connected component with 4 connectivity
    with a depth-first exploration.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccLabel(cv::Mat image)
{
    // 32 int image
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1);
    /********************************************
                YOUR CODE HERE
    *********************************************/

    assert(image.channels() == 1);

    vector<Point2i> neighbours = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    int currentLabel = 0;

    // raster scan
    for (int y = 0; y < image.rows; y++)
    {
        int *resRow = res.ptr<int>(y);

        for (int x = 0; x < image.cols; x++)
        {
            // background or already labeled
            if (!isForeground(image, y, x) || resRow[x] != 0)
                continue;

            // new component
            currentLabel++;
            resRow[x] = currentLabel;

            vector<Point2i> stack;
            stack.push_back(Point2i(x, y));

            while (!stack.empty())
            {
                Point2i r = stack.back();
                stack.pop_back();

                for (const Point2i &n : neighbours)
                {
                    Point2i v = r + n;

                    if (v.x < 0 || v.y < 0 || v.x >= image.cols || v.y >= image.rows)
                        continue;

                    if (res.at<int>(v.y, v.x) == 0 && isForeground(image, v.y, v.x))
                    {
                        res.at<int>(v.y, v.x) = currentLabel;
                        stack.push_back(v);
                    }
                }
            }
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Deletes the connected components (4 connectivity) containg less than size pixels.
*/
cv::Mat ccAreaFilter(cv::Mat image, int size)
{
    Mat res = Mat::zeros(image.rows, image.cols, image.type());
    assert(size > 0);
    /********************************************
                YOUR CODE HERE
    *********************************************/

    assert(image.channels() == 1);

    Mat labels = ccLabel(image);
    unordered_map<int, int> areas;

    // count areas
    for (int y = 0; y < labels.rows; ++y)
    {
        const int *row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            if (row[x] > 0)
                areas[row[x]]++;
        }
    }

    // keep only components with area >= size
    for (int y = 0; y < labels.rows; ++y)
    {
        const int *lrow = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            int label = lrow[x];
            if (label > 0 && areas[label] >= size)
                copyPixel(image, res, y, x);
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccTwoPassLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    /********************************************
                YOUR CODE HERE
    *********************************************/

    assert(image.channels() == 1);

    vector<int> parent;
    parent.push_back(0); // index 0 unused

    auto findRoot = [&](int x)
    {
        // path compression
        int r = x;
        while (parent[r] != r)
            r = parent[r];

        while (parent[x] != x)
        {
            int p = parent[x];
            parent[x] = r;
            x = p;
        }
        return r;
    };

    auto unite = [&](int a, int b)
    {
        int ra = findRoot(a);
        int rb = findRoot(b);
        if (ra != rb)
            parent[rb] = ra;
    };

    int nextLabel = 1;

    // first pass
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            if (!isForeground(image, y, x))
                continue;

            int left = (x > 0) ? res.at<int>(y, x - 1) : 0;
            int up = (y > 0) ? res.at<int>(y - 1, x) : 0;

            if (left == 0 && up == 0)
            {
                res.at<int>(y, x) = nextLabel;
                parent.push_back(nextLabel);
                nextLabel++;
            }
            else
            {
                int minLabel = 0;
                if (left != 0 && up != 0)
                    minLabel = std::min(left, up);
                else
                    minLabel = (left != 0) ? left : up;

                res.at<int>(y, x) = minLabel;

                if (left != 0 && up != 0 && left != up)
                    unite(left, up);
            }
        }
    }

    // second pass: replace by root
    for (int y = 0; y < res.rows; ++y)
    {
        int *row = res.ptr<int>(y);
        for (int x = 0; x < res.cols; ++x)
        {
            if (row[x] > 0)
                row[x] = findRoot(row[x]);
        }
    }

    // canonical relabeling: contiguous labels 1..K in raster order
    unordered_map<int, int> canon;
    int canonNext = 1;

    for (int y = 0; y < res.rows; ++y)
    {
        int *row = res.ptr<int>(y);
        for (int x = 0; x < res.cols; ++x)
        {
            int lab = row[x];
            if (lab == 0)
                continue;

            auto it = canon.find(lab);
            if (it == canon.end())
            {
                canon[lab] = canonNext;
                row[x] = canonNext;
                canonNext++;
            }
            else
            {
                row[x] = it->second;
            }
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
