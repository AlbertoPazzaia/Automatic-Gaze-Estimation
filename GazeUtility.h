#ifndef _GazeUtility_H_
#define _GazeUtility_

#include <opencv2/opencv.hpp>

class GazeUtility
{
public:
	// I detect the gaze of the people
	static void GazeEstimation(cv::Mat& image, cv::Vec3f& pupil_detection)
	{
		// Check to detect an invalid gaze
		if (image.empty())
		{
			pupil_detection = cv::Vec3f(NAN, NAN, NAN);
			return;
		}

		// I create a variable to elaborate the image
		cv::Mat img = image.clone();
		// Initialization of the pupil center position for the gaze pupil
		cv::Point pupil; // estimated pupil centre

		// Preprocessing:
		// Equalization of the image 
		equalizeHist(img, img);
		// I make a thresold 
		cv::threshold(img, img, 60, 255, cv::THRESH_BINARY_INV);
		
		// Test N°1 to verify the binary images:
		//cv::namedWindow("Binary image", cv::WINDOW_NORMAL);
		//cv::imshow("Binary image", img);
		//cv::waitKey(0);

		// Gaussian smoothing
		cv::GaussianBlur(img, img, cv::Size(), 3, 3);

		// Now, I detect the circles with the Hough method:
		// Vector for the iris recognition
		std::vector<cv::Vec3f> iris;

		// HoughCircle method for the circles detection
		cv::HoughCircles(img, iris, cv::HOUGH_GRADIENT, 1, 40, 100, 5, 5, 10);
		// I select as pupil the center of the bigger circle
		pupil = cv::Point(cvRound(iris[0][0]), cvRound(iris[0][1]));

		// Thanks to the Hough Method I select the radius of the Iris circle 
		int radius = cvRound(iris[0][2]);

		// The gaze direction is:
		int gaze_direction;
		// Left
		if (pupil.x < 0.45 * image.cols)
			gaze_direction = 0;
		// Center
		else if (pupil.x >= 0.45 * image.cols && pupil.x <= 0.6 * image.cols)
			gaze_direction = 1;
		//Right
		else
			gaze_direction = 2;

		// I write the position of x and y center and the direction of the gaze
		pupil_detection = cv::Vec3f(pupil.x, pupil.y, gaze_direction);

		// Test N°2 to verify the iris and the pupil position in the image, with respect to a left a right and a central gaze:
		cv::Mat img2 = image.clone();
		cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);
		// I show the pupil
		cv::circle(img2, pupil, 1, CV_RGB(0, 0, 255), 2, cv::LINE_4, 0);
		// threshold 1 and threshol 2 to define the position of the pupil in the image
		double p1 = 0.45 * img2.cols;
		double p2 = 0.6 * img2.cols;
		cv::line(img2, cv::Point(p1, 0), cv::Point(p1, img2.rows), CV_RGB(255, 0, 255), 1, cv::LINE_4, 0);
		cv::line(img2, cv::Point(p2, 0), cv::Point(p2, img2.rows), CV_RGB(255, 0, 255), 1, cv::LINE_4, 0);
		// I show the iris found
		cv::circle(img2, pupil, radius, CV_RGB(0, 0, 255), 1, cv::LINE_8, 0);
		// Decomment if needed
		// cv::namedWindow("Iris and Pupil", cv::WINDOW_NORMAL);
		// cv::imshow("Iris and Pupil", img2);
		// cv::waitKey(0);
	}

	// I show the gazes with an arrow in the image
	static void ArrowImage(cv::Mat& image, std::vector<cv::Vec3f>& left_gazes_v, std::vector<cv::Vec3f>& right_gazes_v, std::vector<cv::Rect>& left_eyes_v, std::vector<cv::Rect>& right_eyes_v)
	{
		int arrow_length = 60;
		// Left eye
		for (int i = 0; i < left_gazes_v.size(); i++)
		{
			// Starting point for the arrow
			cv::Point tail(left_eyes_v[i].x + left_gazes_v[i][0], left_eyes_v[i].y + left_gazes_v[i][1]);
			// Point of the direction
			cv::Point tip;
			if (left_gazes_v[i][2] == 0) // Left
			{
				tip = cv::Point(tail.x - arrow_length, tail.y);
			}
			else if (left_gazes_v[i][2] == 1) // Center
			{
				tip = cv::Point(tail.x, tail.y - arrow_length);
			}
			else if (left_gazes_v[i][2] == 2) //Right
				tip = cv::Point(tail.x + arrow_length, tail.y);
			
			// Arrow
			cv::arrowedLine(image, tail, tip, CV_RGB(255, 0, 0), 2, cv::LINE_8, 0, 0.1);
		}
		//Right eye
		for (int j = 0; j < right_gazes_v.size(); j++)
		{
			// Starting point for the arrow
			cv::Point tail(right_eyes_v[j].x + right_gazes_v[j][0], right_eyes_v[j].y + right_gazes_v[j][1]);
			// Point of the direction
			cv::Point tip;
			if (right_gazes_v[j][2] == 0) // Left
			{
				tip = cv::Point(tail.x - arrow_length, tail.y);
			}
			else if (right_gazes_v[j][2] == 1) // Center
			{
				tip = cv::Point(tail.x, tail.y - arrow_length);
			}
			else if (int(right_gazes_v[j][2]) == 2) //Right
				tip = cv::Point(tail.x + arrow_length, tail.y);
			// Arrow
			cv::arrowedLine(image, tail, tip, CV_RGB(255, 0, 0), 2, cv::LINE_8, 0, 0.1);
		}
	}
};
#endif
