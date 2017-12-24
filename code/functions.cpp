#include "functions.h"


//*****************************************************************************************************
//  ===================================================================================================
//
//	MISC NAMESPACE FUNCTIONS
//
//  ===================================================================================================
//*****************************************************************************************************

//-----------------------------------------------------------------------------------------------------
/*
   Function: misc::Round
   Round number to the nearest integer
   Parameters:
      d - float
   Returns:
      int result

*/
//-----------------------------------------------------------------------------------------------------
float aia::vesselsegment::misc::Round(float d)
{
	return (float)floor(d + 0.5);
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: misc::PrintAccuracy
   Print accuracy of each image in the dataset
   Parameters:
      accuracy_rate - a vector of double numbers
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::misc::PrintAccuracy(std::vector<double> & accuracy_rate)
{
	int image_num=0;
	for (std::vector<double>::const_iterator i = accuracy_rate.begin(); i != accuracy_rate.end(); ++i)
	{
		image_num++;
		double acc = *i;
		cout << "Image " << to_string(image_num) << " : " << acc*100  << "%" << endl;
	}
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: misc::GetImagesInFolder
   Get all images in a folder and save them to a vector of Mat
   Parameters:
      folder		- folder directory of images, masks or groundtruths
	  ext			- format of image, for example, .tif, .png, .jpg
	  force_gray	- read image as gray if true
   Returns:
      vector of Mat (images)
*/
//-----------------------------------------------------------------------------------------------------
std::vector <cv::Mat> aia::vesselsegment::misc::GetImagesInFolder(std::string folder, std::string ext, 
		bool force_gray) throw (aia::error)
{
	// get all files within folder
	std::vector < std::string > files;
	cv::glob(folder, files);

	// open files that contains 'ext'
	std::vector < cv::Mat > images;
	for(auto & f : files)
	{
		if(f.find(ext) == std::string::npos)
			continue;

		cv::Mat img = cv::imread(f, force_gray ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_UNCHANGED);
		images.push_back(img);
	}

	return images;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: misc::ConvertBinary
   Convert an image to binary
   Parameters:
      im_in		-	input image
	  im_out	- 	output image
	  threshold	-	threshold to binarize the image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::misc::ConvertBinary(Mat & im_in, Mat & im_out, double threshold)
{
	im_out = im_in.clone();

	for (int i = 0; i < im_in.cols; i++ ) {
		for (int j = 0; j < im_in.rows; j++) {
			if (im_in.at<float>(j, i) > threshold) {   
				im_out.at<float>(j, i) = 1.0;
			}
			else
				im_out.at<float>(j, i) = 0;
		}
	}
}




//*****************************************************************************************************
//  ===================================================================================================
//
//	MORPHOLOGY NAMESPACE FUNCTIONS
//
//  ===================================================================================================
//*****************************************************************************************************

//-----------------------------------------------------------------------------------------------------
/*
   Function: morphology::FindEndPoint
   Find ending points of skeleton image
   Parameters:
      im_in		-	input image
   Returns:
      vector of Point2i which are endpoints of skeleton image
*/
//-----------------------------------------------------------------------------------------------------
std::vector<cv::Point2i> aia::vesselsegment::morphology::FindEndPoint (Mat & im_in)
{
	// Declare variable to count neighbourhood pixels
	int count;

	// To store a pixel intensity
	float pix;

	// To store the ending co-ordinates
	std::vector<Point2i> coords;

	// For each pixel in our image...
	for (int i = 1; i < im_in.rows-1; i++) 
	{
		for (int j = 1; j < im_in.cols-1; j++) 
		{
			// See what the pixel is at this location
			pix = im_in.at<float>(i,j);

			// If not a skeleton point, skip
			if (pix == 0)
				continue;

			// Reset counter
			count = 0;     

			// For each pixel in the neighbourhood
			// centered at this skeleton location...
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {

					// Get the pixel in the neighbourhood
					pix = im_in.at<float>(i+y,j+x);

					// Count if non-zero
					if (pix != 0)
						count++;
				}
			}

			// If count is exactly 2, add co-ordinates to vector
			if (count == 2) 
			{
				cv::Point point = Point(j,i);
				coords.push_back(point);
			}
		}
	}
	return coords;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: morphology::FindJunctionPoint
   Find junction points of skeleton image
   Parameters:
      im_in		-	input image
   Returns:
      vector of Point2i which are junction points of skeleton image
*/
//-----------------------------------------------------------------------------------------------------
std::vector<cv::Point2i> aia::vesselsegment::morphology::FindJunctionPoint (Mat & im_in)
{
	// Declare variable to count neighbourhood pixels
	int count;

	// To store a pixel intensity
	float pix;

	// To store the ending co-ordinates
	std::vector<Point2i> coords;

	// For each pixel in our image...
	for (int i = 1; i < im_in.rows-1; i++) 
	{
		for (int j = 1; j < im_in.cols-1; j++) 
		{

			// See what the pixel is at this location
			pix = im_in.at<float>(i,j);

			// If not a skeleton point, skip
			if (pix == 0)
				continue;

			// Reset counter
			count = 0;     

			// For each pixel in the neighbourhood
			// centered at this skeleton location...
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {

					// Get the pixel in the neighbourhood
					pix = im_in.at<float>(i+y,j+x);

					// Count if non-zero
					if (pix != 0)
						count++;
				}
			}

			// If count is exactly 2, add co-ordinates to vector
			if (count > 3) 
			{
				cv::Point point = Point(j,i);
				coords.push_back(point);
			}
		}
	}
	return coords;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: morphology::ThinSubiteration1
   Thin sub-function for Morphology thinning
   Parameters:
      im_in		-	input image
	  im_out	-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::morphology::ThinSubiteration1(Mat & im_in, Mat & im_out) {
	int rows = im_in.rows;
	int cols = im_in.cols;
	im_in.copyTo(im_out);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			if(im_in.at<float>(i, j) == 1.0f) {
				/// get 8 neighbors
				/// calculate C(p)
				int neighbor0 = (int) im_in.at<float>( i-1, j-1);
				int neighbor1 = (int) im_in.at<float>( i-1, j);
				int neighbor2 = (int) im_in.at<float>( i-1, j+1);
				int neighbor3 = (int) im_in.at<float>( i, j+1);
				int neighbor4 = (int) im_in.at<float>( i+1, j+1);
				int neighbor5 = (int) im_in.at<float>( i+1, j);
				int neighbor6 = (int) im_in.at<float>( i+1, j-1);
				int neighbor7 = (int) im_in.at<float>( i, j-1);
				int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
					int(~neighbor3 & ( neighbor4 | neighbor5)) +
					int(~neighbor5 & ( neighbor6 | neighbor7)) +
					int(~neighbor7 & ( neighbor0 | neighbor1));
				if(C == 1) {
					/// calculate N
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1,N2);
					if ((N == 2) || (N == 3)) {
						/// calculate criteria 3
						int c3 = ( neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
						if(c3 == 0) {
							im_out.at<float>( i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: morphology::ThinSubiteration1
   Thin sub-function for Morphology thinning
   Parameters:
      im_in		-	input image
	  im_out	-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::morphology::ThinSubiteration2(Mat & im_in, Mat & im_out) {
	int rows = im_in.rows;
	int cols = im_in.cols;
	im_in.copyTo( im_out);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			if (im_in.at<float>( i, j) == 1.0f) {
				/// get 8 neighbors
				/// calculate C(p)
				int neighbor0 = (int) im_in.at<float>( i-1, j-1);
				int neighbor1 = (int) im_in.at<float>( i-1, j);
				int neighbor2 = (int) im_in.at<float>( i-1, j+1);
				int neighbor3 = (int) im_in.at<float>( i, j+1);
				int neighbor4 = (int) im_in.at<float>( i+1, j+1);
				int neighbor5 = (int) im_in.at<float>( i+1, j);
				int neighbor6 = (int) im_in.at<float>( i+1, j-1);
				int neighbor7 = (int) im_in.at<float>( i, j-1);
				int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
					int(~neighbor3 & ( neighbor4 | neighbor5)) +
					int(~neighbor5 & ( neighbor6 | neighbor7)) +
					int(~neighbor7 & ( neighbor0 | neighbor1));
				if(C == 1) {
					/// calculate N
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1,N2);
					if((N == 2) || (N == 3)) {
						int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
						if(E == 0) {
							im_out.at<float>(i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: morphology::MorphologyThinning
   Morphology thinning
   Parameters:
      im_in		-	input image
	  im_out	-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::morphology::MorphologyThinning(Mat & im_in, Mat & im_out) {
	bool bDone = false;
	int rows = im_in.rows;
	int cols = im_in.cols;

	Mat image = im_in.clone();
	image.convertTo(image,CV_32FC1, 255);

	image.copyTo(im_out);

	im_out.convertTo(im_out,CV_32FC1);

	/// pad source
	Mat p_enlarged_src = Mat(rows + 2, cols + 2, CV_32FC1);
	for(int i = 0; i < (rows+2); i++) {
		p_enlarged_src.at<float>(i, 0) = 0.0f;
		p_enlarged_src.at<float>( i, cols+1) = 0.0f;
	}
	for(int j = 0; j < (cols+2); j++) {
		p_enlarged_src.at<float>(0, j) = 0.0f;
		p_enlarged_src.at<float>(rows+1, j) = 0.0f;
	}
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			if (image.at<float>(i, j) >= 20.0f) {
				p_enlarged_src.at<float>( i+1, j+1) = 1.0f;
			}
			else
				p_enlarged_src.at<float>( i+1, j+1) = 0.0f;
		}
	}

	/// start to thin
	Mat p_thinMat1 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_thinMat2 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_cmp = Mat::zeros(rows + 2, cols + 2, CV_8UC1);

	while (bDone != true) {
		/// sub-iteration 1
		aia::vesselsegment::morphology::ThinSubiteration1(p_enlarged_src, p_thinMat1);
		/// sub-iteration 2
		aia::vesselsegment::morphology::ThinSubiteration2(p_thinMat1, p_thinMat2);
		/// compare
		compare(p_enlarged_src, p_thinMat2, p_cmp, CV_CMP_EQ);
		/// check
		int num_non_zero = countNonZero(p_cmp);
		if(num_non_zero == (rows + 2) * (cols + 2)) {
			bDone = true;
		}
		/// copy
		p_thinMat2.copyTo(p_enlarged_src);
	}
	// copy result
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			im_out.at<float>( i, j) = p_enlarged_src.at<float>( i+1, j+1);
		}
	}
}




//*****************************************************************************************************
//  ===================================================================================================
//
//	LINE OPERATOR NAMESPACE FUNCTIONS
//
//  ===================================================================================================
//*****************************************************************************************************


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::GenerateRange
   Linspace in Matlab for int number
   Parameters:
      a	-	starting number
	  b	-	step
	  c	-	end number
   Returns:
	  vector of int number  	
*/
//-----------------------------------------------------------------------------------------------------
std::vector<int> aia::vesselsegment::lineoperator::GenerateRange(int a, int b, int c) {
	vector<int> array;
	while(a <= c) {
		array.push_back(a);
		a += b;         // could recode to better handle rounding errors
	}
	return array;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::ComputeMeanAndStd
   Compute mean and std of an array
   Parameters:
      v		- array of float number
   Returns:	
	  mean	- mean of array
	  stdev	- standard deviation of array
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::lineoperator::ComputeMeanAndStd (vector<float> & v, float & mean, float & stdev)
{
	float sum = std::accumulate(v.begin(), v.end(), 0.0);
	mean = sum / v.size();

	float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
	stdev = std::sqrt(sq_sum / v.size() - mean * mean);
	stdev = std::sqrt(pow(stdev,2)*v.size()/(v.size()-1));
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::ExtractNonZeroElement
   Extract in-mask region from an image and stack all elements into a vector
   Parameters:
      im_in		-	input image
	  mask		-	mask image
   Returns:	
	  vector of float elements
*/
//-----------------------------------------------------------------------------------------------------
vector<float> aia::vesselsegment::lineoperator::ExtractNonZeroElement (Mat & im_in, Mat & mask)
{
	vector<float> v;
	for (int i = 0; i < im_in.cols; i++ ) {
		for (int j = 0; j < im_in.rows; j++) {
			if (mask.at<float>(j, i) > 0) {   
				//cout << i << ", " << j << endl;     // Do your operations
				float element = im_in.at<float>(j, i);
				v.push_back(element);
			}
		}
	}
	return v;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::GetPoints
   Get start and end points to draw a line in line structure
   Parameters:
      size		-	size of kernel
	  theta		-	angle between the line and horizontal axis in counter-clockwise
   Returns:	
	  point1	-	starting point
	  point2	-	end point
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::lineoperator::GetPoints (int size, int theta, cv::Point & point1, cv::Point & point2)
{
	int x1, y1, x2, y2;
	int halfsize = (size-1)/2;
	if (theta == 0)
	{
		x1 = 0; y1 = halfsize;
		x2 = size-1; y2 = halfsize;
	}
	else if (theta == 90)
	{
		y1 = 0; x1 = halfsize;
		y2 = size-1; x2 = halfsize;
	}
	else 
	{
		if (theta > 90)
			theta = 180-theta;
		x1 = -halfsize;	
		y1 = aia::vesselsegment::misc::Round(x1*(sin(theta*PI/180)/cos(theta*PI/180)));
		if (y1 < -halfsize)
		{
			y1 = -halfsize;
			x1 = aia::vesselsegment::misc::Round(y1*(cos(theta*PI/180)/sin(theta*PI/180)));
		}
		x2 = halfsize;
		y2 = aia::vesselsegment::misc::Round(x2*(sin(theta*PI/180)/cos(theta*PI/180)));
		if (y2 > halfsize)
		{
			y2 = halfsize;
			x2 = aia::vesselsegment::misc::Round(y2*(cos(theta*PI/180)/sin(theta*PI/180)));
		}
		y1 = halfsize-y1;
		x1 = halfsize+x1;
		y2 = halfsize-y2;
		x2 = halfsize+x2;
	}
	point1 = cv::Point(x1, y1);
	point2 = cv::Point(x2, y2);
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::CreateLineStructure
   Line structure
   Parameters:
      L					-	size of kernel
	  theta_array_size	-	number of theta which we want to create 0,15,30,45....
   Returns:	
	  vector of Mat (line kernels)
*/
//-----------------------------------------------------------------------------------------------------
vector<Mat> aia::vesselsegment::lineoperator::CreateLineStructure(int L, int theta_array_size) throw (ucas::Error)
{
	// check preconditions
	if( L%2 == 0 )
		throw ucas::Error(ucas::strprintf("Structuring element L (%d) is not odd", L));

	cv::Point point1, point2;
	cv::vector <cv::Mat> SEs;
	double angle_step = 15;
	for(int k=0; k<theta_array_size; k++)
	{
		cv::Mat SE = Mat::zeros(L, L, CV_32F);
		string s = "line of theta = " + to_string(k*15);
		//cout << s << endl;
		float theta = angle_step*k;
		if (theta>90)
		{
			theta = theta - 90;
			aia::vesselsegment::lineoperator::GetPoints (L, theta, point1, point2);
			cv::line(SE, point1, point2, cv::Scalar(1.0));
			cv::warpAffine(SE, SE, cv::getRotationMatrix2D(cv::Point2f((SE.cols-1)/2, (SE.rows-1)/2), 90, 1.0), cv::Size(L, L), CV_INTER_NN);
		}
		else
		{
			aia::vesselsegment::lineoperator::GetPoints (L, theta, point1, point2);
			cv::line(SE, point1, point2, cv::Scalar(1.0));
		}
		SEs.push_back(SE);
	}
	return SEs;	 
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::GetLineaResponse
   Apply line structure filter to obtain line response
   Parameters:
      im_in		-	input image
	  W			-	kernel size (in this project we set it as 15 - max width of vessel)
	  L			-	line-structure size
   Returns:	
	  im_out	-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::lineoperator::GetLineaResponse(Mat & im_in, Mat & im_out, int W, int L)
{
	Mat image = im_in.clone();
	double min, max;
	minMaxLoc(image, &min, &max);
	Mat avgresponse;
	// new method
	cv::Point anchor = cv::Point(-1,-1); double delta = 0; int ddepth = -1;
	Mat SE2 = Mat::ones(W,W,CV_32F);
	SE2 = SE2/W/W;
	filter2D(image, avgresponse, ddepth , SE2, anchor, delta, BORDER_DEFAULT ); // correct

	vector<int> theta_array = aia::vesselsegment::lineoperator::GenerateRange(0, 15, 165);
	cv::vector<cv::Mat>	SEs = aia::vesselsegment::lineoperator::CreateLineStructure(L,theta_array.size());

	Mat kernel;
	Mat imglinestrength;
	Mat maxlinestrength = Mat::ones(image.size(), image.type());
	maxlinestrength = maxlinestrength*(-100);

	for (int i=0; i<theta_array.size(); i++)
	{
		Mat SE = SEs[i];
		SE = SE/L;
		filter2D(image, imglinestrength, ddepth , SE, anchor, delta, BORDER_DEFAULT );
		string window_name = "imglinestrength at " + to_string(i);
		imglinestrength = imglinestrength - avgresponse;
		maxlinestrength = cv::max(maxlinestrength,imglinestrength);    

	}
	im_out =  maxlinestrength.clone();
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::NormalizeImage
   Normalize region of image inside the mask (=(image-mean)/stdev)
   Parameters:
      im_in		-	input image
	  mask		-	mask
   Returns:	
	  im_out	-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::lineoperator::NormalizeImage(Mat & image, Mat & mask, Mat & im_out)
{
	double mina, maxa;
	minMaxLoc(image, &mina, &maxa);

	im_out = image.clone();
	vector<float> v = aia::vesselsegment::lineoperator::ExtractNonZeroElement (image, mask);
	size_t size_v = v.size();

	float mean, stdev, element;
	aia::vesselsegment::lineoperator::ComputeMeanAndStd (v, mean, stdev);

	for (int i = 0; i < image.cols; i++ ) {
		for (int j = 0; j < image.rows; j++) {
			if (mask.at<float>(j, i) > 0) {   
				element = (image.at<float>(j, i)-mean)/stdev;
				im_out.at<float>(j, i) = element;
			}
			else
				im_out.at<float>(j, i) = 0;
		}
	}

}


//-----------------------------------------------------------------------------------------------------
/*
   Function: lineoperator::LineOperatorSegment
   Segment an image using line operator
   Parameters:
      im_in		-	input image
	  mask		-	mask
	  features	-	line response
	  W			-	kernel size (in this project we set it as 15 - max width of vessel)
	  step		-	step to increase in line structure 1,3,5,7,...
	  threshold	-	threshold to binarize image
   Returns:	
	  segmented image using line operator method
*/
//-----------------------------------------------------------------------------------------------------
Mat aia::vesselsegment::lineoperator::LineOperatorSegment(Mat & im_in, Mat & mask, 
		Mat & features, int W, int step, double threshold)
{
	Mat input, im_out;
	input = im_in.clone(); // correct

	Mat sub_mat = Mat::ones(im_in.size(), im_in.type());
	subtract(sub_mat, input, input);

	Vector<float> v = aia::vesselsegment::lineoperator::ExtractNonZeroElement (input, mask);
	aia::vesselsegment::lineoperator::NormalizeImage(input, mask, features);

	vector<int> Ls = aia::vesselsegment::lineoperator::GenerateRange(1,step,W);

	Mat R;
	for (int i=0; i<Ls.size(); i++ )

	{
		int L = Ls[i];

		aia::vesselsegment::lineoperator::GetLineaResponse(input, R, W, L); // correct

		aia::vesselsegment::lineoperator::NormalizeImage(R, mask, R);

		features = features + R;
	}

	features = features/(Ls.size()+1);
	//imshow("features", features);

	double min, max;
	minMaxLoc(features, &min, &max);
	/*cout << "min is: " << min << endl;
	cout << "max is: " << max << endl;*/

	cv::threshold(features, im_out, threshold, 1, cv::THRESH_BINARY);

	/*imshow("final", im_out);*/

	return im_out;
}


////-----------------------------------------------------------------------------------------------------
///*
//   Function: lineoperator::LineOperatorAllImages
//   Segment all images at one time using line operator and save them to features
//   Parameters:
//      im_in		-	input image
//	  mask		-	mask
//	  features	-	line response
//	  W			-	kernel size (in this project we set it as 15 - max width of vessel)
//	  step		-	step to increase in line structure 1,3,5,7,...
//	  threshold	-	threshold to binarize image
//   Returns:	
//	  segmented image using line operator method
//*/
////-----------------------------------------------------------------------------------------------------
//void aia::vesselsegment::lineoperator::LineOperatorAllImages
//		(std::vector <cv::Mat> & images, std::vector <cv::Mat> & truths, 
//		std::vector <cv::Mat> & masks, std::vector <cv::Mat> & features_images)
//{
//	cout << "erosing masks" << endl;
//	masks = aia::vesselsegment::evaluation::ErodeMasks (masks);
//
//	for (int i=0; i<images.size(); i++)
//		/*for (int i=0; i<1; i++)*/
//	{
//		string s = "image " + to_string(i+1);
//		cout << s << endl;
//
//		// read image, truth, mask
//		cv::Mat image = images[i];
//		cv::Mat truth = truths[i];
//		cv::Mat mask = masks[i];
//		/*mask.convertTo(mask, CV_32F, 1.0/255);
//		ConvertBinary(mask, mask, 0.5);
//		ErodeMask (mask, mask);*/
//
//		// Split the image into different channels
//		cv::Mat bgr[3];   //destination array
//		split(image,bgr);//split source  
//
//		cv::Mat image_blue = bgr[0];
//		cv::Mat image_green = bgr[1];
//		cv::Mat image_red = bgr[2];
//
//		/*image_green.convertTo(image_green, CV_32F, 1.0/255);*/
//		cv::normalize(image_green, image_green, 0, 1.0, cv::NORM_MINMAX, CV_32F);
//		cv::normalize(truth, truth, 0, 1.0, cv::NORM_MINMAX, CV_32F);
//		cv::normalize(mask, mask, 0, 1.0, cv::NORM_MINMAX, CV_32F);
//		Mat input = image_green.clone();
//		//Mat input = image_green.clone();
//		Mat features;
//		Mat final = ImageSegment(input, mask, features, 15, 2, SEGMENT_THRESHOLD);
//		features_images.push_back(features);
//	}
//}




//*****************************************************************************************************
//  ===================================================================================================
//
//	REMOVE REGION NAMESPACE FUNCTIONS
//
//  ===================================================================================================
//*****************************************************************************************************

//-----------------------------------------------------------------------------------------------------
/*
   Function: removeregion::NormalizeImage
   Find blobs (or regions) by 8-connectivity
   Parameters:
      binary	-	input binary image
   Returns:	
	  blobs		-	vector of vector of points which represents all blobs in a binary image 
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::removeregion::FindBlob(const cv::Mat & binary, 
		std::vector < std::vector<cv::Point2i> > & blobs)
{
	blobs.clear();

	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground

	cv::Mat label_image;
	binary.convertTo(label_image, CV_32SC1);

	int label_count = 2; // starts at 2 because 0,1 are used already

	for(int y=0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		for(int x=0; x < label_image.cols; x++) {
			if(row[x] != 1) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 8);

			std::vector <cv::Point2i> blob;

			for(int i=rect.y; i < (rect.y+rect.height); i++) 
			{
				int *row2 = (int*)label_image.ptr(i);
				for(int j=rect.x; j < (rect.x+rect.width); j++) 
				{
					if(row2[j] != label_count) {
						continue;
					}

					blob.push_back(cv::Point2i(j,i));
				}
			}

			blobs.push_back(blob);

			label_count++;
		}
	}
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: removeregion::RemoveSmallRegion
   Remove small blobs (or regions) which has size smaller than a threshold
   Parameters:
      im_in		-	input image
	  minSize	-	threshold to remove the small regions
   Returns:	
	  im_out	-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::removeregion::RemoveSmallRegion (Mat & im_in, Mat & im_out, int minSize)
{
	Mat input = im_in.clone();
	//input.convertTo(input, CV_8U, 255);
	threshold(input, input, 0.0, 1.0, THRESH_BINARY);


	cv::Mat result = cv::Mat::zeros(input.size(), CV_32F);

	std::vector < std::vector<cv::Point2i > > blobs;


	// find all blobs in binary image
	aia::vesselsegment::removeregion::FindBlob(input, blobs);

	// for each blob, if blob size > minSize -> store, if not ignore
	for(size_t i=0; i < blobs.size(); i++) 
	{
		if (blobs[i].size()>minSize)
		{
			for(size_t j=0; j < blobs[i].size(); j++) 
			{
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;
				result.at<float>(y,x) = 1;
			}
		}

	}

	// return image out
	im_out = result.clone();
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: removeregion::GetNumberJuncpointInBlob
   Count number of junction (branch) points in a blob - used to remove zigzag region
   Parameters:
      blob			-	vector of points which represents one blob in a binary image
	  juncpoints	-	vector of Point2i which are junction points of skeleton image
   Returns:	
	  count			-	number of junction (branch) points in a blob
*/
//-----------------------------------------------------------------------------------------------------
int aia::vesselsegment::removeregion::GetNumberJuncpointInBlob (vector<cv::Point2i> blob, vector<Point2i> juncpoints)
{
	int count = 0;
	for (int i=0; i<blob.size(); i++)
	{
		Point2i point_blob = blob[i];
		for (int j=0; j<juncpoints.size(); j++)
		{
			Point2i point_junc = juncpoints[j];
			if (point_blob == point_junc)
				count++;
		}
	}
	return count;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: removeregion::RemoveZigzagRegion
   Remove zigzag regions
   Parameters:
      im_in					-	input image
	  area_threshold		-	first condition - if a blob having size smaller than this threshold, 
								we will consider to remove if it satisfies second condition
	  branchpoint_threshold	-	second condition - if a blob containing more branchpoints than this
								threshold then remove
   Returns:	
	  im_out				-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::removeregion::RemoveZigzagRegion (Mat & im_in, 
		Mat & im_out, int area_threshold, int branchpoint_threshold)
{
	Mat image = im_in.clone();
	Mat regions, big_regions, small_regions, small_regions_skeleton;


	aia::vesselsegment::removeregion::RemoveSmallRegion (image, big_regions, area_threshold);
	small_regions = image-big_regions;

	aia::vesselsegment::morphology::MorphologyThinning(small_regions, small_regions_skeleton);
	vector<Point2i> juncpoints = aia::vesselsegment::morphology::FindJunctionPoint (small_regions_skeleton);

	vector <vector<cv::Point2i>> blobs;
	aia::vesselsegment::removeregion::FindBlob(small_regions, blobs);

	Mat zigzag_regions = Mat::zeros(image.size(), image.type());
	vector<Point2i> blob;
	int N;
	for(size_t i=0; i < blobs.size(); i++) 
	{
		blob = blobs[i];
		N = aia::vesselsegment::removeregion::GetNumberJuncpointInBlob (blob, juncpoints);

		if  (N > branchpoint_threshold)
		{
			for(size_t j=0; j < blob.size(); j++) 
			{
				int x = blob[j].x;
				int y = blob[j].y;
				zigzag_regions.at<float>(y,x) = 1;
			}
		}
	}

	im_out = im_in - zigzag_regions;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: removeregion::FindCenterOfOptic
   Estimate center of optic disk
   Parameters:
      im_in		-	input image
	  W			-	radius of optic disk (100)
   Returns:	
	  Point21	-	represents an estimated center of optic disk 
*/
//-----------------------------------------------------------------------------------------------------
Point2i aia::vesselsegment::removeregion::FindCenterOfOptic (Mat & im_in, int W)
{
	Mat image = im_in.clone(); 
	double minVal, maxVal;
	Point2i minLoc, maxLoc;
	Mat avgresponse;
	// new method
	cv::Point anchor = cv::Point(-1,-1); double delta = 0; int ddepth = -1;
	Mat SE2 = Mat::ones(W,W,CV_32F);
	SE2 = SE2/W/W;
	filter2D(image, avgresponse, ddepth , SE2, anchor, delta, BORDER_DEFAULT ); // correct
	Mat img8bit;
	avgresponse.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "opticdisk.jpg", img8bit );
	minMaxLoc(avgresponse, &minVal, &maxVal, &minLoc, &maxLoc );
	return maxLoc;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: removeregion::RemoveRegionsMadeByOpticDisk
   Region noisy regions made by optic disk
   Parameters:
      im_in				-	input image
	  im_seg			-	binary image (output of line operator)
	  W					-	radius of optic disk (100)
	  optic_threshold	-	threshold to remove noisy pixels made by optic disk
   Returns:	
	  im_out			-	output image
   Algorithm:
	  + Step 1: determine optic disk region
	  + Step 2: extract ROI of segmented image and original image
	  + Step 3: compute top-hat transform of ROI original image
	  + Step 4: multiply pixel-wise result of step 3 and ROI segemented image
	  + Step 5: remove pixel in ROI if smaller than a threshold
*/
//-----------------------------------------------------------------------------------------------------
Mat aia::vesselsegment::removeregion::RemoveRegionsMadeByOpticDisk(Mat & im_in, Mat & im_seg, int W, int optic_threshold)
{
	Mat image = im_in.clone(); 
	Point2i maxLoc = aia::vesselsegment::removeregion::FindCenterOfOptic (image, W);
	int x_topleft = max(maxLoc.x-(optic_threshold-1)/2, 0);
	int y_topleft = max(maxLoc.y-(optic_threshold-1)/2, 0);
	int x_bottomright = min(maxLoc.x+(optic_threshold-1)/2, im_in.cols);
	int y_bottomright = min(maxLoc.y+(optic_threshold-1)/2, im_in.rows);

	Point2i topLeft = Point2i (x_topleft, y_topleft);
	Point2i bottomRight = Point2i (x_bottomright, y_bottomright);

	//Create a rect
	Rect R(topLeft,bottomRight);  

	// Extract ROI
	Mat seg_ROI = im_seg(R);
	Mat img_ROI = im_in(R);


	// Create a structuring element (SE)
	int morph_size = 1;
	Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

	Mat img8bit;

	Mat image_tophat_ROI;
	Mat sub_mat = Mat::ones(img_ROI.size(), img_ROI.type());
	subtract(sub_mat, img_ROI, img_ROI);

	// Apply the tophat morphology operation
	for (int i=1;i<10;i++)
	{   
		morphologyEx(img_ROI, image_tophat_ROI, MORPH_TOPHAT, element, Point(-1,-1), i );   
	}


	for (int i = 0; i < seg_ROI.cols; i++ ) {
		for (int j = 0; j < seg_ROI.rows; j++) {
			if (!seg_ROI.at<float>(j, i)) {   
				image_tophat_ROI.at<float>(j,i) = 0;
			}
		}
	}

	img_ROI.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "img_ROI.jpg", img8bit );

	image_tophat_ROI.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "image_tophat_ROI.jpg", img8bit );

	seg_ROI.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "seg_ROI.jpg", img8bit );


	// Remove optic disk noise
	Mat ROI_mul = image_tophat_ROI.mul(seg_ROI);

	ROI_mul.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "ROI_mul.jpg", img8bit );

	Mat seg_ROI_removed_optic = Mat::zeros(image_tophat_ROI.size(), image_tophat_ROI.type());

	for (int i = 0; i < image_tophat_ROI.cols; i++ ) {
		for (int j = 0; j < image_tophat_ROI.rows; j++) {
			if (image_tophat_ROI.at<float>(j, i) >OPTICDISK_THRESHOLD)
				seg_ROI_removed_optic.at<float>(j,i) = 1;
		}
	}

	seg_ROI_removed_optic.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "ROI_removed.jpg", img8bit );


	// Remove weird regions after removing noisy optic disk
	aia::vesselsegment::removeregion::RemoveSmallRegion (seg_ROI_removed_optic, seg_ROI_removed_optic, OPTICDISK_SMALL_THRESHOLD);
	aia::vesselsegment::removeregion::RemoveZigzagRegion (seg_ROI_removed_optic, 
		seg_ROI_removed_optic, OPTICDISK_AREA_THRESHOLD, OPTICDISK_BRANCHPOINT_THRESHOLD);


	Mat im_out = im_seg.clone();
	seg_ROI_removed_optic.copyTo(im_out(R));
	return im_out;
}




//*****************************************************************************************************
//  ===================================================================================================
//
//	FILL HOLE NAMESPACE FUNCTIONS
//
//  ===================================================================================================
//*****************************************************************************************************

//-----------------------------------------------------------------------------------------------------
/*
   Function: fillhole::FillHole
   Fill all holes in a binary image
   Parameters:
      im_in		-	input image
   Returns:	
	  im_out	-	output image
	  holes		-	holes image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::fillhole::FillHole (Mat & im_in, Mat & im_out, Mat & holes)
{
	//assume input is uint8 B & W (0 or 1)
	//this function imitates imfill(image,'hole')
	im_out = im_in.clone();
	cv::Mat filled=im_in.clone();
	cv::floodFill(filled,cv::Point2i(0,0),cv::Scalar(1));

	for (int i = 1; i < filled.rows-1; i++) 
	{
		for (int j = 1; j < filled.cols-1; j++) 
		{
			if (!filled.at<float>(i,j))
				im_out.at<float>(i,j) = 1;
		}
	}
	holes = im_out - im_in;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: fillhole::FillSmallHole
   Fill all small holes which have size smaller than a threshold
   Parameters:
      im_in		-	input image
	  hole_size -	maximum size of a hole to be filled
   Returns:	
	  im_out	-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::fillhole::FillSmallHole (Mat& im_in, Mat& im_out, int hole_size)
{
	Mat holes, image_filled_all_holes, big_holes, small_holes;
	Mat image = im_in.clone();
	aia::vesselsegment::fillhole::FillHole (image, image_filled_all_holes, holes);
	aia::vesselsegment::removeregion::RemoveSmallRegion (holes, big_holes, hole_size);
	small_holes = holes - big_holes;
	im_out = image + small_holes;
}




//*****************************************************************************************************
//  ===================================================================================================
//
//	REGION GROWING NAMESPACE FUNCTIONS - IN PROGRESS (NOT GIVING POSITIVE RESULT)
//
//  ===================================================================================================
//*****************************************************************************************************

//-----------------------------------------------------------------------------------------------------
/*
   Function: growregion::IsBlobConnectMainVessel
   Check if a blob is connected to main vessle
   Parameters:
      im_in		-	input image
	  blob		-	vector of points which represents one blob in a binary image
   Returns:	
	  bool		-	true if connected, false otherwise
*/
//-----------------------------------------------------------------------------------------------------
bool aia::vesselsegment::growregion::IsBlobConnectMainVessel (Mat & im_in, std::vector<cv::Point2i> blob)
{
	int count = 0;
	for(size_t i=0; i< blob.size(); i++) 
	{
		int x0 = blob[i].x;
		int y0 = blob[i].y;


		// For each pixel in the neighbourhood
		// centered at this skeleton location...
		for (int y = -1; y <= 1; y++) 
		{
			for (int x = -1; x <= 1; x++) 
			{
				float pix = im_in.at<float>(y0+y,x0+x);

				// Count if non-zero
				if (pix != 0)
					count++;
			}
		}
	}
	if (count)
		return true;
	else
		return false;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: growregion::IsPointConnectMainVessel
   Check if a point is connected to main vessle
   Parameters:
      im_in		-	input image
	  point		-	a point to check if it is connected to main vessel
   Returns:	
	  bool		-	true if connected, false otherwise
*/
//-----------------------------------------------------------------------------------------------------
bool aia::vesselsegment::growregion::IsPointConnectMainVessel (Mat& im_in, cv::Point2i point)
{
	int count = 0;

	int x0 = point.x;
	int y0 = point.y;

	// For each pixel in the neighbourhood
	// centered at this skeleton location...
	for (int y = -1; y <= 1; y++) 
	{
		for (int x = -1; x <= 1; x++) 
		{
			float pix = im_in.at<float>(y0+y,x0+x);

			// Count if non-zero
			if (pix != 0)
				count++;
		}
	}

	if (count)
		return true;
	else
		return false;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: growregion::RegionGrowing
   Region growing - NOT SUCCESSFUL (NOTE!!!! Think about connect to endpoint only)
   Parameters:
      im_in				-	input image
	  features			-	line response image
	  grow_threshold	-	threshold to grow 
   Returns:	
	  im_out			-	output image after growing
*/
//-----------------------------------------------------------------------------------------------------
Mat aia::vesselsegment::growregion::RegionGrowing(Mat & im_in, Mat & features, float grow_threshold)
{
	Mat image = im_in.clone();
	Mat result;
	cv::threshold(features, result, grow_threshold, 1, cv::THRESH_BINARY);
	result = result - image;

	// Extend if connected to main vessel
	std::vector < std::vector<cv::Point2i> > blobs;

	aia::vesselsegment::removeregion::RemoveSmallRegion (result, result, 5);

	aia::vesselsegment::removeregion::FindBlob(result, blobs);

	std::vector<cv::Point2i> blob;

	Mat image_grow = Mat::zeros(image.size(),image.type());

	for (int i=0; i<blobs.size(); i++)
	{
		blob = blobs[i];
		if (aia::vesselsegment::growregion::IsBlobConnectMainVessel (image, blob)) // if connected to main vessel
		{
			for(size_t j=0; j < blob.size(); j++) 
			{
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;
				image_grow.at<float>(y,x) = 1;
			}
		}

	}

	Mat im_out = image + image_grow;

	return im_out;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: growregion::RegionPatch
   Region patching - if a point having intensity greater than a threshold, then add it to final result
   Parameters:
      im_in				-	input image
	  features			-	line response image
	  grow_threshold	-	threshold to grow 
   Returns:	
	  im_out			-	output image after growing
*/
//-----------------------------------------------------------------------------------------------------
Mat aia::vesselsegment::growregion::RegionPatch(Mat & im_in, Mat & features, float grow_threshold)
{
	Mat image = im_in.clone();
	Mat image_regions_grow;
	cv::threshold(features, image_regions_grow, grow_threshold, 1, cv::THRESH_BINARY);
	image_regions_grow = image_regions_grow - image;

	// Remove small regions
	Mat image_grow = Mat::zeros(image.size(),image.type());

	// For each pixel in our image...
	for (int i = 1; i < image_regions_grow.rows-1; i++) 
	{
		for (int j = 1; j < image_regions_grow.cols-1; j++) 
		{
			if (image_regions_grow.at<float>(i,j))
			{
				cv::Point2i point = Point2i(j,i);
				if (aia::vesselsegment::growregion::IsPointConnectMainVessel (image, point))
					image_grow.at<float>(i,j) = 1;
			}
		}
	}
	image_grow = image+image_grow;

	return image_grow;
}




//*****************************************************************************************************
//  ===================================================================================================
//
//	EVALUATION NAMESPACE FUNCTIONS
//
//  ===================================================================================================
//*****************************************************************************************************

//-----------------------------------------------------------------------------------------------------
/*
   Function: evaluation::ComputeAccuracy
   Compute accuracy of all images
   Parameters:
      segmented_images	-	segmented images
	  grounds			-	groundtruth images
	  masks				-	mask images (only consider pixel inside the mask)
	  accuracy_array	-	array indicates each image's accuracy
   Returns:	
	  double number		-	accuracy for all images
*/
//-----------------------------------------------------------------------------------------------------
double aia::vesselsegment::evaluation::ComputeAccuracy(std::vector <cv::Mat> & segmented_images, 
		std::vector <cv::Mat> & grounds, std::vector <cv::Mat> & masks, std::vector<double> & accuracy_array) 
{
	// True positives (TP), True negatives (TN), and total number N of pixels are all we need
	double TP = 0, TN = 0, N = 0;

	// examine one image at the time
	for(size_t i=0; i<segmented_images.size(); i++)
	{
		// the caller did not ask to calculate visual results
		// accuracy calculation is easier...
		cv::Mat seg = segmented_images[i];
		cv::normalize(seg, seg, 0, 1.0, cv::NORM_MINMAX, CV_32F);
		cv::Mat ground = grounds[i];
		cv::normalize(ground, ground, 0, 1.0, cv::NORM_MINMAX, CV_32F);
		cv::Mat mask = masks[i];
		cv::normalize(mask, mask, 0, 1.0, cv::NORM_MINMAX, CV_32F);

		//cv::Mat seg = segmented_images[i];
		//seg.convertTo(seg, CV_32F);
		//cv::Mat ground = grounds[i];
		//ground.convertTo(ground, CV_32F, 1.0/255);
		//cv::Mat mask = masks[i];
		//mask.convertTo(mask, CV_32F);

		double TP_image = 0,  TN_image = 0, N_image = 0;

		for(int y=0; y<seg.rows; y++)
		{
			for(int x=0; x<seg.cols; x++)
			{
				if(mask.at<float>(y, x))
				{
					N++;		// found a new sample within the mask
					N_image++;

					if(seg.at<float>(y,x) && ground.at<float>(y,x))
					{
						TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)
						TP_image++;
					}
					else if(!seg.at<float>(y,x) && !ground.at<float>(y,x))
					{
						TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)
						TN_image++;
					}	
				}
			}

		}

		double accuracy = (TP_image+TN_image)/N_image;
		accuracy_array.push_back(accuracy);
	}
	return (TP + TN) / N;	// according to the definition of Accuracy
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: evaluation::ComputeAccuracyOneImage
   Compute accuracy of one image
   Parameters:
      seg				-	segmented image
	  ground			-	groundtruth image
	  mask				-	mask image (only consider pixel inside the mask)
   Returns:	
	  double number		-	accuracy for one image
*/
//-----------------------------------------------------------------------------------------------------
double aia::vesselsegment::evaluation::ComputeAccuracyOneImage(cv::Mat& seg, cv::Mat& ground, cv::Mat& mask) 
{
	// examine one image at the time

	// the caller did not ask to calculate visual results
	// accuracy calculation is easier...
	
	seg.convertTo(seg, CV_32F);
	ground.convertTo(ground, CV_32F, 1.0/255);
	mask.convertTo(mask, CV_32F);

	double TP_image = 0,  TN_image = 0, N_image = 0;

	for(int y=0; y<seg.rows; y++)
	{
		for(int x=0; x<seg.cols; x++)
		{
			if(mask.at<float>(y, x))
			{
				N_image++;

				if(seg.at<float>(y,x) && ground.at<float>(y,x))
				{
					TP_image++;
				}
				else if(!seg.at<float>(y,x) && !ground.at<float>(y,x))
				{
					TN_image++;
				}	
			}
		}

	}

	double accuracy = (TP_image+TN_image)/N_image;
	return accuracy;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: evaluation::CompareImages
   Compare 2 images and display the differences
   Parameters:
      img1				-	image 1
	  img1				-	image 2
	  s					-	string to display
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::evaluation::CompareImages(Mat & img1, Mat & img2, string s)
{
	std::vector<cv::Mat> channels;
	cv::Mat imgPair;
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);
	channels.push_back(img2);
	channels.push_back(img1);
	channels.push_back(img2);
	merge(channels, imgPair);
	cv::imshow(s, imgPair);

	Mat img8bit;
	imgPair.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite(s, img8bit );
	
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: evaluation::ErodeMask
   Erode mask by n (=5) pixel using circle filter to avoid weird outer circle
   Parameters:
      im_in				-	input image
   Returns:	
	  im_out			-	output image
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::evaluation::ErodeMask (Mat& im_in, Mat& im_out)
{
	cv::Mat sel = (cv::Mat_<uchar>(9,9) <<	0, 0, 1, 1, 1, 1, 1, 0, 0,
		0, 1, 1, 1, 1, 1, 1, 1, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 1, 1, 1, 1, 1, 1, 1, 0,
		0, 0, 1, 1, 1, 1, 1, 0, 0);
	erode(im_in, im_out, sel); 
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: evaluation::ErodeMask
   Erode all masks by n (=5) pixel using circle filter to avoid weird outer circle
   Parameters:
      masks				-	input images
   Returns:	
	  new masks			-	output images
*/
//-----------------------------------------------------------------------------------------------------
vector<Mat> aia::vesselsegment::evaluation::ErodeMasks (vector <Mat> masks)
{
	vector <Mat> masks_eroded;
	size_t N = masks.size();
	for (int i=0; i<N; i++)
	{
		Mat mask=masks[i];
		/*mask.convertTo(mask, CV_32F, 1.0/255);*/
		cv::normalize(mask, mask, 0, 1.0, cv::NORM_MINMAX, CV_32F);
		aia::vesselsegment::misc::ConvertBinary(mask, mask, 0.5);
		aia::vesselsegment::evaluation::ErodeMask (mask, mask);
		masks_eroded.push_back(mask);
	}
	return masks_eroded;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: evaluation::ProcessOneImage
   Process one image
   Parameters:
      images			-	original images
	  grounds			-	groundtruth images
	  masks				-	mask images (only consider pixel inside the mask)
	  image_number		-	image number
   Returns:	
	  segmented_image	-	final result
*/
//-----------------------------------------------------------------------------------------------------
Mat aia::vesselsegment::evaluation::ProcessOneImage(std::vector <cv::Mat> images, 
		std::vector <cv::Mat> truths, std::vector <cv::Mat> masks, int image_number)
{
	// read image, truth, mask
	cv::Mat image = images[image_number];
	cv::Mat truth = truths[image_number];
	cv::Mat mask = masks[image_number];

	mask.convertTo(mask, CV_32F, 1.0/255);
	aia::vesselsegment::misc::ConvertBinary(mask, mask, 0.5);
	aia::vesselsegment::evaluation::ErodeMask (mask, mask);

	// Split the image into different channels
	cv::Mat bgr[3];   //destination array
	split(image,bgr);//split source  

	cv::Mat image_blue = bgr[0];
	cv::Mat image_green = bgr[1];
	cv::Mat image_red = bgr[2];

	image_green.convertTo(image_green, CV_32F, 1.0/255);
	Mat input = image_green.clone();

	//// apply the CLAHE algorithm to the L channel
	//cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	//clahe->setClipLimit(4);
	//cv::Mat dst;
	//clahe->apply(image_green, dst);
	//dst.convertTo(dst, CV_32F, 1.0/255);
	//Mat input = dst.clone();


	cv::imshow("Original image", input);

	Mat features;
	Mat image_seg = aia::vesselsegment::lineoperator::LineOperatorSegment(input, mask, features, 15, 2, 0.6);

	cv::imshow("features", features);
	string s = "image " + to_string(image_number+1);
	cv::imshow(s, image_seg);

	Mat img8bit;
	input.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "input.jpg", img8bit );

	features.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "features.jpg", img8bit);

	image_seg.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "image_seg.jpg", img8bit);

	// Fill small holes
	Mat image_filled_small_holes;
	aia::vesselsegment::fillhole::FillSmallHole (image_seg, image_filled_small_holes, HOLE_THRESHOLD);
	cv::imshow("After fill holes", image_filled_small_holes);

	image_filled_small_holes.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "image_filled_small_holes.jpg", img8bit);

	// Remove small regions
	Mat image_removed_small_regions;
	aia::vesselsegment::removeregion::RemoveSmallRegion (image_filled_small_holes, image_removed_small_regions, BLOB_THRESHOLD);
	cv::imshow("After remove small regions", image_removed_small_regions);

	image_removed_small_regions.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "image_removed_small_regions.jpg", img8bit);

	// Remove zigzag regions
	Mat image_removed_zigzag_regions;
	aia::vesselsegment::removeregion::RemoveZigzagRegion (image_removed_small_regions, image_removed_zigzag_regions, AREA_THRESHOLD, BRANCHPOINT_THRESHOLD);
	cv::imshow("After remove zigzag regions", image_removed_zigzag_regions);

	image_removed_zigzag_regions.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "image_removed_zigzag_regions.jpg", img8bit);

	// Region patching
	Mat image_region_patch = aia::vesselsegment::growregion::RegionPatch(image_removed_zigzag_regions, features, SEGMENT_THRESHOLD-0.03);
	cv::imshow("After patch regions", image_region_patch);

	// Region growing
	Mat image_region_grow = aia::vesselsegment::growregion::RegionGrowing(image_region_patch, features, GROW_THRESHOLD);
	cv::imshow("After grow regions", image_region_grow);

	aia::vesselsegment::evaluation::CompareImages(image_region_grow, image_removed_zigzag_regions, "image_region_grow.jpg");


	// Remove noisy optic-disk region
	Mat image_removed_optic;
	image_removed_optic = aia::vesselsegment::removeregion::RemoveRegionsMadeByOpticDisk(input, image_removed_zigzag_regions, 15, 201);
	cv::imshow("After remove optic disk noise", image_removed_optic);

	image_removed_optic.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "image_removed_optic.jpg", img8bit);

	Mat final = image_removed_optic.clone();

	double accuracy = aia::vesselsegment::evaluation::ComputeAccuracyOneImage(final, truth, mask);
	std::cout << "Accuracy is: " << accuracy << endl;

	aia::vesselsegment::evaluation::CompareImages(image_removed_optic, truth, "final_truth.jpg");

	return final;
}


//-----------------------------------------------------------------------------------------------------
/*
   Function: evaluation::ProcessAllImage
   Process all images
   Parameters:
      images			-	original images
	  grounds			-	groundtruth images
	  masks				-	mask images (only consider pixel inside the mask)
   Returns:	
	  segmented_images	-	segmented images
*/
//-----------------------------------------------------------------------------------------------------
void aia::vesselsegment::evaluation::ProcessAllImages(std::vector <cv::Mat> images, std::vector <cv::Mat> truths, 
		std::vector <cv::Mat> masks, std::vector <cv::Mat> segmented_images)
{
	masks = aia::vesselsegment::evaluation::ErodeMasks (masks);

	for (int i=0; i<images.size(); i++)
		/*for (int i=0; i<1; i++)*/
	{
		// read image, truth, mask
		cv::Mat image = images[i];
		cv::Mat truth = truths[i];
		cv::Mat mask = masks[i];
		/*mask.convertTo(mask, CV_32F, 1.0/255);
		aia::vesselsegment::misc::ConvertBinary(mask, mask, 0.5);
		aia::vesselsegment::evaluation::ErodeMask (mask, mask);*/

		// Split the image into different channels
		cv::Mat bgr[3];   //destination array
		split(image,bgr);//split source  

		cv::Mat image_blue = bgr[0];
		cv::Mat image_green = bgr[1];
		cv::Mat image_red = bgr[2];


		/*image_green.convertTo(image_green, CV_32F, 1.0/255);*/
		cv::normalize(image_green, image_green, 0, 1.0, cv::NORM_MINMAX, CV_32F);
		cv::normalize(truth, truth, 0, 1.0, cv::NORM_MINMAX, CV_32F);
		cv::normalize(mask, mask, 0, 1.0, cv::NORM_MINMAX, CV_32F);
		Mat input = image_green.clone();
		//Mat input = image_green.clone();
		Mat features;
		Mat final = aia::vesselsegment::lineoperator::LineOperatorSegment(input, mask, features, 15, 2, SEGMENT_THRESHOLD);

		string s = "image " + to_string(i+1);
		/*imshow(s, final);*/

		cout << "======" << s << endl;


		// Fill small holes
		aia::vesselsegment::fillhole::FillSmallHole (final, final, HOLE_THRESHOLD);

		// Remove small regions
		aia::vesselsegment::removeregion::RemoveSmallRegion (final, final, BLOB_THRESHOLD);

		// Remove zigzag regions
		aia::vesselsegment::removeregion::RemoveZigzagRegion (final, final, AREA_THRESHOLD, BRANCHPOINT_THRESHOLD);

		// Remove optic disk
		final = aia::vesselsegment::removeregion::RemoveRegionsMadeByOpticDisk(input, final, 15, 201);

		//// Grow regions
		/*final = aia::vesselsegment::growregion::RegionGrowing(final, features, GROW_THRESHOLD);*/

		// Region patching
		/*final = aia::vesselsegment::growregion::RegionPatch(final, features, SEGMENT_THRESHOLD-0.03);*/


		stringstream ss;

		string name = "image_";
		string type = ".jpg";

		ss<<name<<(i+1)<<type;

		string filename = ss.str();
		/*ss.str("");*/

		/*imwrite(filename, img_cropped);*/


		aia::vesselsegment::evaluation::CompareImages(final, truth, filename);



		// Save in final results
		segmented_images.push_back(final);

	}

	double acc;
	vector<double> accuracy_rate;
	acc = aia::vesselsegment::evaluation::ComputeAccuracy(segmented_images, truths, masks, accuracy_rate);
	aia::vesselsegment::misc::PrintAccuracy(accuracy_rate);
	cout << "Total accuracy is: " << 100*acc << "%" << endl;
}


////-----------------------------------------------------------------------------------------------------
///*
//   Function: evaluation::GetPointsRocCurve
//   Process all images
//   Parameters:
//      images			-	original images
//	  grounds			-	groundtruth images
//	  masks				-	mask images (only consider pixel inside the mask)
//   Returns:	
//	  segmented_images	-	segmented images
//*/
////-----------------------------------------------------------------------------------------------------
//void aia::vesselsegment::evaluation::GetPointsRocCurve(std::vector <cv::Mat> images, std::vector <cv::Mat> truths, 
//					std::vector <cv::Mat> masks)
//{
//	std::vector <cv::Mat> segmented_images;
//		std::vector <cv::Mat> segmented_images_post;
//		std::vector <cv::Mat> features_images;
//		vector <vector<double>> R, R_post;
//
//		// =============================================================================================================
//		// All images
//		// =============================================================================================================
//		aia::vesselsegment::lineoperator::LineOperatorAllImages(images, truths, masks, features_images);
//
//		for (double threshold = -1; threshold <= 8; threshold = threshold+0.01)
//		{
//			// clear for another loop
//			segmented_images.clear();
//			segmented_images_post.clear();
//
//			// compute features
//			aia::vesselsegment::lineoperator::SegmentAllImages(images, segmented_images, segmented_images_post, 
//				features_images, threshold);
//
//			// Save to vector
//			vector<double> accuracy_array_post;
//			std::vector<double> scores_post = aia::vesselsegment::ComputeAccuracyAndValuation(segmented_images_post, truths, masks, accuracy_array_post);
//			scores_post.push_back(threshold);
//			R_post.push_back(scores_post);
//
//			vector<double> accuracy_array;
//			std::vector<double> scores = aia::vesselsegment::ComputeAccuracyAndValuation(segmented_images, truths, masks, accuracy_array);
//			scores.push_back(threshold);
//			R.push_back(scores);
//
//
//			cout << "============================================================" << endl;
//			cout << "threshold: " << to_string(threshold) << endl;
//			cout << "accuracy | false positive rate | true positive rate" << endl;
//			cout << "------------------------------------------------------------" << endl;
//			cout << "Line operator:" << endl;
//			cout << scores[0] << " | " << scores[1] << " | " << scores[2] << endl;
//			cout << "------------------------------------------------------------" << endl;
//			cout << "Post processing:" << endl;
//			cout << scores_post[0] << " | " << scores_post[1] << " | " << scores_post[2] << endl;
//
//		}
//
//		/*double threshold = SEGMENT_THRESHOLD;*/
//
//
//		ofstream myfile_scores;
//		myfile_scores.open ("scores.txt");
//		for (int i=0; i<R.size(); i++)
//		{
//			myfile_scores << R[i][0] << " " << R[i][1] << " " << R[i][2] << " " << R[i][3] << endl;
//		}
//		myfile_scores.close();
//
//
//		ofstream myfile_scores_post;
//		myfile_scores_post.open ("scores_post.txt");
//		for (int i=0; i<R.size(); i++)
//		{
//			myfile_scores_post << R_post[i][0] << " " << R_post[i][1] << " " << R_post[i][2] << " " << R_post[i][3] << endl;
//		}
//		myfile_scores_post.close();
//}
