#ifndef _project_0_h
#define _project_0_h

// include aia and ucas utility functions
#include "aia/aiaConfig.h"
#include "ucas/ucasConfig.h"


// include my project libs
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <strsafe.h>
#include <numeric>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "fstream"
#include "iostream"
#include <windows.h>
#include <Shlwapi.h>
#include <filesystem> // C++17 (or Microsoft-specific implementation in C++14)


// include my project libs
using namespace cv;
using namespace std;



// define Constants
#define PI 3.14159265
#define SEGMENT_THRESHOLD 0.57				// threshold for line operator
#define BLOB_THRESHOLD 30					// threshold for removing blobs 
#define HOLE_THRESHOLD 10					// threshold for filling holes			
#define AREA_THRESHOLD 150					// threshold for removing zigzag regions
#define BRANCHPOINT_THRESHOLD 0				// threshold for removing zigzag regions
#define GROW_THRESHOLD 0.6					// threshold for growing regions
#define GROW_BLOB_THRESHOLD 40				// threshold for removing small regions after growing regions
#define GROW_AREA_THRESHOLD 70				// threshold for removing zigzag regions after growing regions
#define GROW_BRANCHPOINT_THRESHOLD 0		// threshold for removing zigzag regions after growing regions
#define OPTICDISK_THRESHOLD 0.045			// threshold for removing optic disk
#define OPTICDISK_SMALL_THRESHOLD 10		// threshold for removing small regions after removing optic disk
#define OPTICDISK_AREA_THRESHOLD 70			// threshold for removing zigzag regions after removing optic disk
#define OPTICDISK_BRANCHPOINT_THRESHOLD 0	// threshold for removing zigzag regions after removing optic disk


namespace aia
{
	// namespace projecy vessel sementation
	namespace vesselsegment
	{
		// misc functions
		namespace misc
		{
			float Round(float d);
			std::vector <cv::Mat> GetImagesInFolder(std::string folder, std::string ext = ".tif", bool force_gray = false) throw (aia::error); // get all images from 1 folder
			void PrintAccuracy(vector<double>& accuracy_rate);
			void ConvertBinary(Mat& input, Mat& im_out, double thresholdBin);
		}
		
		// morphology functions
		namespace morphology
		{
			vector<cv::Point2i> FindEndPoint (Mat& im);
			vector<cv::Point2i> FindJunctionPoint (Mat & im);
			void ThinSubiteration1(Mat & im_in, Mat & im_out);
			void ThinSubiteration2(Mat & im_in, Mat & im_out);
			void MorphologyThinning(Mat & im_in, Mat & im_out);
		}

		// remove region functions
		namespace removeregion
		{
			void FindBlob(const cv::Mat & binary, std::vector <std::vector<cv::Point2i>> & blobs);
			void RemoveSmallRegion (Mat & im_in, Mat& im_out, int minSize);
			int GetNumberJuncpointInBlob (vector<cv::Point2i> blob, vector<Point2i> juncpoints);
			void RemoveZigzagRegion (Mat & im_in, Mat & im_out, int area_threshold, int branchpoint_threshold);
			Point2i FindCenterOfOptic (Mat & im_in, int W);
			Mat RemoveRegionsMadeByOpticDisk(Mat & im_in, Mat & im_seg, int W, int optic_threshold);
		}

		// line operator functions
		namespace lineoperator
		{
			void ComputeMeanAndStd (vector<float> & v, float & mean, float & stdev);
			vector<float> ExtractNonZeroElement (Mat & image, Mat & mask);
			void GetPoints (int size, int theta, cv::Point& point1, cv::Point& point2);
			vector<Mat> CreateLineStructure(int L, int n) throw (ucas::Error);
			void GetLineaResponse(Mat& input, Mat& R, int W, int L);
			vector<int> GenerateRange(int a, int b, int c);
			void NormalizeImage(Mat& image, Mat& mask, Mat& im_out);
			Mat LineOperatorSegment(Mat & image, Mat & mask, Mat & features, int W, int step, double thresholdBin);
			void LineOperatorAllImages(std::vector <cv::Mat> & images, std::vector <cv::Mat> & truths, 
					std::vector <cv::Mat> & masks, std::vector <cv::Mat> & features_images);
		}

		// grow region functions
		namespace growregion
		{
			bool IsBlobConnectMainVessel (Mat& im_in, std::vector<cv::Point2i> blob);
			bool IsPointConnectMainVessel (Mat& im_in, cv::Point2i point);
			Mat RegionGrowing(Mat & im_in, Mat & features, float grow_threshold);
			Mat RegionPatch(Mat & im_in, Mat & features, float grow_threshold);
		}

		// fill hole functions
		namespace fillhole
		{
			void FillHole(Mat & im_in, Mat & im_out, Mat & holes);
			void FillSmallHole (Mat & im_in, Mat & im_out, int hole_size);
		}

		// evaluation functions
		namespace evaluation
		{
			double ComputeAccuracy(std::vector <cv::Mat> & segmented_images, std::vector <cv::Mat> & grounds, 
					std::vector <cv::Mat> & masks, std::vector<double> & accuracy_array);
			double ComputeAccuracyOneImage(cv::Mat& seg, cv::Mat& ground, cv::Mat& mask); 
			void CompareImages(Mat& img1, Mat& img2, string s);
			void ErodeMask (Mat& input, Mat& im_out);
			vector <Mat> ErodeMasks (vector <Mat> masks);
			Mat ProcessOneImage(std::vector <cv::Mat> images, std::vector <cv::Mat> truths, 
					std::vector <cv::Mat> masks, int image_number);
			void ProcessAllImages(std::vector <cv::Mat> images, std::vector <cv::Mat> truths, 
					std::vector <cv::Mat> masks, std::vector <cv::Mat> segmented_images);
			void GetPointsRocCurve(std::vector <cv::Mat> images, std::vector <cv::Mat> truths, 
					std::vector <cv::Mat> masks);
		}

	}
}

#endif // _project_0_h