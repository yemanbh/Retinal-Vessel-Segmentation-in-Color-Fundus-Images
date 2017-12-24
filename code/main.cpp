#include "functions.h"


//-----------------------------------------------------------------------------------------------------
//	Main
//-----------------------------------------------------------------------------------------------------
int main() 
{
	try
	{
		std::string dataset_path = "C:/Users/RD/Google Drive/Working/Retinal/Matlab/dataset/STARE/";


		std::vector <cv::Mat> images = aia::vesselsegment::misc::GetImagesInFolder(dataset_path + "images", ".ppm", false);
		std::vector <cv::Mat> truths = aia::vesselsegment::misc::GetImagesInFolder(dataset_path + "groundtruth", ".ppm", true);
		std::vector <cv::Mat> masks  = aia::vesselsegment::misc::GetImagesInFolder(dataset_path + "mask", ".ppm", true);



		//// declare dataset path
		//std::string dataset_path = "C:/Users/RD/Google Drive/Working/Retinal/Matlab/dataset/";
		///*std::string dataset_path = "C:/Users/RD/Google Drive/Master MAIA/Second Semester/Image Analyist/Projects/AIA-Retinal-Vessel-Segmentation/dataset/DRIVE/test/";*/

		//// read all images 	
		//std::vector <cv::Mat> images = aia::vesselsegment::misc::GetImagesInFolder(dataset_path + "images", ".tif", false);
		//std::vector <cv::Mat> truths = aia::vesselsegment::misc::GetImagesInFolder(dataset_path + "groundtruth", ".tif", true);
		//std::vector <cv::Mat> masks  = aia::vesselsegment::misc::GetImagesInFolder(dataset_path + "mask", ".tif", true);

		// declare segmented images
		std::vector <cv::Mat> segmented_images;


		int option = 2; // 1: one image, 2 all images, 3 write AUC

		int image_number = 2;



		if (option==1)
		{
			aia::vesselsegment::evaluation::ProcessOneImage(images, truths, masks, image_number-1);
		}
		else if (option==2) 
		{
			aia::vesselsegment::evaluation::ProcessAllImages(images, truths, masks, segmented_images);
		}
		else
		{
		}

		cvWaitKey(0);
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}