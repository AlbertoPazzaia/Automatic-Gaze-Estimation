#include <opencv2/opencv.hpp> // I used opencv-4.0.1
#include <iostream>

#include "FacesEyesUtility.h"
#include "GazeUtility.h"

int main(int argc, char* argv[]) {
	// I read the images from the folder "images"
	std::vector<cv::String> fn;
	cv::String path("..\\images\\");
	cv::glob(path, fn, false);
	// Index of the image used for the preprocessing
	int image_index;

	// Cicle to read all the images
	for (image_index = 1; image_index <= fn.size(); image_index++)
	{
		// Definition of the color and gray images:
		cv::Mat color_image;
		cv::Mat gray_image;
		// Acquisition of the color image
		color_image = cv::imread(path + std::to_string(image_index) + ".jpg", cv::IMREAD_COLOR);
		// Check for invalid input
		if (color_image.empty())
		{
			std::cout << "Error reading the image." << std::endl;
			return -1;
		}

		// Preprocessing of the images with a poor resolution, to enforce eye and face recognition
		// I use Bicubic interpolation with different parameters for every image, because even if it is slower it gives better results
		if (image_index == 1) // Image 1
		{
			cv::resize(color_image, color_image, cv::Size(), 1, 1, cv::INTER_CUBIC);
		}
		else if (image_index == 2) // Image 2
		{
			cv::resize(color_image, color_image, cv::Size(), 1.5, 1.5, cv::INTER_CUBIC);
		}
		if (image_index == 4 || image_index == 14) // Same parameters for images 4 and 14
		{
			cv::resize(color_image, color_image, cv::Size(), 2.5, 2.5, cv::INTER_CUBIC);
		}
		if (image_index == 8) // For the decimation of image 8 I use Lanczos interpolation
		{
			cv::resize(color_image, color_image, cv::Size(), 0.5, 0.5, cv::INTER_LANCZOS4);
		}
		else if (image_index == 13 || image_index == 15) // Same parameters for images 5 and 15
		{
			cv::resize(color_image, color_image, cv::Size(), 2, 2, cv::INTER_CUBIC);
		}

		// Convertion for better processing
		cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);

		// We show the source images after the resize
		cv::namedWindow("Input resized image", cv::WINDOW_NORMAL);
		cv::imshow("Input resized image", color_image);
		cv::waitKey(0);

		// If needed here we have the gray images
		// cv::namedWindow("Grayscale image", cv::WINDOW_NORMAL);
		// cv::imshow("Grayscale image", gray_image);
		// cv::waitKey(0);

		// Here I detect and display the images and the eyes
		// Initialization of the vectors of the faces, of the left eyes and of the right eyes. Then a Mat to show the image with the faces and the eyes we found
		std::vector<cv::Rect> faces;
		std::vector<cv::Rect> left_eyes;
		std::vector<cv::Rect> right_eyes;
		cv::Mat F_E_D_color_image = color_image.clone(); // Face and eyes detected color image

		//Faces and eyes detection
		FacesEyesUtility::FacesEyesLocalization(gray_image, faces, left_eyes, right_eyes, F_E_D_color_image);

		// After the division of left and right eyes here I estimate the gaze of every eye
		// Gaze estimation of the left eyes
		std::vector<cv::Vec3f> left_eyes_gazes;
		for (int i = 0; i < left_eyes.size(); i++)
		{
			// Vector for the estimations
			cv::Vec3f vector_temp;
			// Gray image of every left eye for the estimations
			cv::Mat image_temp = gray_image(left_eyes[i]);

			GazeUtility::GazeEstimation(image_temp, vector_temp);
			// Push to store the vector of gaze estimations
			left_eyes_gazes.push_back(vector_temp);
		}
		// Gaze estimation of the right eyes
		std::vector<cv::Vec3f> right_eyes_gazes; // vector of the right gazes 
		for (int j = 0; j < right_eyes.size(); j++)
		{
			// Vector for the estimations
			cv::Vec3f vector_temp;
			// Gray image of every right eye for the estimations
			cv::Mat image_temp = gray_image(right_eyes[j]);

			GazeUtility::GazeEstimation(image_temp, vector_temp);
			right_eyes_gazes.push_back(vector_temp);
		}

		// Now I show the image computed in the gaze estimation
		cv::Mat estimated_image = color_image.clone();
		GazeUtility::ArrowImage(estimated_image, left_eyes_gazes, right_eyes_gazes, left_eyes, right_eyes);
		cv::namedWindow("Gaze estimation", cv::WINDOW_NORMAL);
		cv::imshow("Gaze estimation", estimated_image);
		cv::waitKey(0);

		// To make everything more clear i destroy the windows every time
		cv::destroyAllWindows();
	}
	return 0;
}
