#ifndef _FacesEyesUtility_H_
#define _FacesEyesUtility_

#include <opencv2/opencv.hpp>

class FacesEyesUtility
{
public:
	// I detect faces and eys positions and I show them in the original image
	static void FacesEyesLocalization(cv::Mat& image, std::vector<cv::Rect>& faces_v, std::vector<cv::Rect>& left_v, std::vector<cv::Rect>& right_v, cv::Mat& F_E_D_color_img)
	{
		// I load the Cascade Classifier for the faces
		cv::CascadeClassifier face_Cascade_Classifier;
		std::string face_Classifier = "haarcascade_frontalface_alt.xml";
		bool load_test1 = face_Cascade_Classifier.CascadeClassifier::load(face_Classifier);
		// I verify the correctness of the loading
		if (!load_test1)
		{
			std::cout << "Error loading the face Cascade Classifier" << std::endl;
			return;
		}
		// We detect the faces and save them
		face_Cascade_Classifier.CascadeClassifier::detectMultiScale(image, faces_v, 1.01, 3, 0, cv::Size(100, 100), image.size()); // scaleFactor = 1.01 & minNeighbors = 3
		std::vector<cv::Mat> faces_imgs;
		for (int i = 0; i < faces_v.size(); i++)
		{
			cv::Mat image_temp = image(faces_v[i]);
			faces_imgs.push_back(image_temp);
		}

		// I load the Cascade Classifier for the eyes
		cv::CascadeClassifier eyes_Cascade_Classifier;
		std::string eyes_Classifier = "haarcascade_eye_tree_eyeglasses.xml";
		// I verify the correctness of the loading
		bool load_test2 = eyes_Cascade_Classifier.CascadeClassifier::load(eyes_Classifier);
		if (!load_test2)
		{
			std::cout << "Error loading the eyes Cascade Classifier" << std::endl;
			return;
		}
		// We detect the eyes, anayze if they are really eyes then and save them in the left or in the right eyes vector
		for (int i = 0; i < faces_v.size(); i++)
		{
			// Vectors used for the operations of eyes classification
			std::vector<cv::Rect> eyes_temp;
			std::vector<cv::Rect> eyes_temp_left;
			std::vector<cv::Rect> eyes_temp_right;
			eyes_Cascade_Classifier.CascadeClassifier::detectMultiScale(faces_imgs[i], eyes_temp, 1.01, 4, 0, cv::Size(30, 30), cv::Size(75, 75)); // scaleFactor = 1.01 & minNeighbors = 4

			// For the first classification I select the correct eyes: I consider the ones in the first half of the faces and I take one left/right eye per face
			for (int j = 0; j < eyes_temp.size(); j++)
			{
				// x coordinate of the eye
				int eye_x_coord = eyes_temp[j].x + eyes_temp[j].width / 2;
				// y coordinate of the eye
				int c_y_eye = eyes_temp[j].y + eyes_temp[j].height / 2;
				// I check with the coodinate if the eyes are in the first half of the faces
				if (c_y_eye < faces_imgs[i].rows / 2)
				{
					// I divide right eyes from left eyes saving them in the correct vector
					if (eye_x_coord < faces_imgs[i].cols / 2)
						eyes_temp_left.push_back(eyes_temp[j]);
					else
						eyes_temp_right.push_back(eyes_temp[j]);
				}
			}

			// Now, I select the correct eyes from all the eyes retrived and I place them in the left temp vector
			// If I have more then one left eye in a face
			if (eyes_temp_left.size() > 1)
			{
				// I select those more distant ones from the nose
				std::vector<int> eyes_x;
				for (int j = 0; j < eyes_temp_left.size(); j++)
					eyes_x.push_back(eyes_temp_left[j].x);
				int min_eyes_x_index;
				cv::minMaxIdx(eyes_x, NULL, NULL, &min_eyes_x_index, NULL);
				left_v.push_back(eyes_temp_left[min_eyes_x_index]);
			}
			// If I have one left eye in a face
			else if (eyes_temp_left.size() == 1)
			{
				left_v.push_back(eyes_temp_left[0]);
			}
			// If I do not have any left eye in a face I insert an empty one
			else
				left_v.push_back(cv::Rect());
			// Now the right eyes
			// If I have more then one right eye in a face
			if (eyes_temp_right.size() > 1)
			{
				// I select those more distant ones from the nose
				std::vector<int> eyes_x;
				for (int j = 0; j < eyes_temp_right.size(); j++)
					eyes_x.push_back(eyes_temp_right[j].x);
				int max_eyes_x_index;
				cv::minMaxIdx(eyes_x, NULL, NULL, NULL, &max_eyes_x_index);
				right_v.push_back(eyes_temp_right[max_eyes_x_index]);
			}
			// If I have one right eye in a face
			else if (eyes_temp_right.size() == 1)
			{
				right_v.push_back(eyes_temp_right[0]);
			}
			// If I do not have any right eye in a face I insert an empty one
			else
				right_v.push_back(cv::Rect());
		}

		// For the second classification I select the faces with at least one eye, to be sure I do not take wrong images
		// Vectors used for the operations of face classification
		std::vector<cv::Rect> faces_temp;
		std::vector<cv::Rect> face_left_eye;
		std::vector<cv::Rect> face_right_eye;
		for (int j = 0; j < faces_v.size(); j++) // fcs.size() is the total number of faces
			// If the face has at least one left or right eye I save the image and the respective eye
			if (!left_v[j].empty() || !right_v[j].empty())
			{
				faces_temp.push_back(faces_v[j]);
				face_left_eye.push_back(left_v[j]);
				face_right_eye.push_back(right_v[j]);
			}
		faces_v = faces_temp;
		left_v = face_left_eye;
		right_v = face_right_eye;

		// Now, I plot the faces and the eyes in the resized image. I shift the coordinates of the eyes in every face 
		for (int i = 0; i < faces_v.size(); i++)
		{
			cv::Point eyes_shift(faces_v[i].x, faces_v[i].y);
			left_v[i] = left_v[i] + eyes_shift;
			right_v[i] = right_v[i] + eyes_shift;
		}
		for (int j = 0; j < faces_v.size(); j++)
		{
			// Ellipse on the face
			cv::Point2d face_center(faces_v[j].x + faces_v[j].width * 0.5, faces_v[j].y + faces_v[j].height * 0.5);
			cv::ellipse(F_E_D_color_img, face_center, cv::Size(faces_v[j].width * 0.4, faces_v[j].height * 0.6), 0, 0, 360, cv::Scalar(255, 0, 255), 4, cv::LINE_4, 0);
			// Ellipse on the left eye
			cv::Point2d left_eye_center(left_v[j].x + left_v[j].width * 0.5, left_v[j].y + left_v[j].height * 0.5);
			cv::ellipse(F_E_D_color_img, left_eye_center, cv::Size(left_v[j].width * 0.5, left_v[j].height * 0.5), 0, 0, 360, cv::Scalar(255, 0, 0), 4, cv::LINE_4, 0);
			// Ellipse on the right eye
			cv::Point2d right_eyes_center(right_v[j].x + right_v[j].width * 0.5, right_v[j].y + right_v[j].height * 0.5);
			cv::ellipse(F_E_D_color_img, right_eyes_center, cv::Size(right_v[j].width * 0.5, right_v[j].height * 0.5), 0, 0, 360, cv::Scalar(255, 0, 0), 4, cv::LINE_4, 0);
		}
		// Print
		cv::namedWindow("Faces and Eyes detected", cv::WINDOW_NORMAL);
		cv::imshow("Faces and Eyes detected", F_E_D_color_img);
		cv::waitKey(0);
	}
};
#endif
