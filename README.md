# Automatic Gaze Estimation
In this algorithm, I use Computer Vision to identify a person's face and the direction they are looking in. To do this I use c++ and OpenCV (version [4.0.1](https://opencv.org/release/opencv-4-0-1/)). Future versions of [OpenCV](https://opencv.org/releases/) may improve the algorithm's capabilities but require code changes.

### Instructions:
In this project, the objective is to locate the eyes in the images provided and then analyze the regions to find the iris and estimate the direction of the gaze. The estimate is divided into three categories: looking straight, looking right, and looking left.
After a pre-processing and a face and eyes recognition, I used some Computer Vision algorithms to perform the gaze estimation.

### Acquisition and Pre-processing:
The images are loaded from a predefined "images" folder and are read through a loop, as they came in ascending alphanumeric order from 1 to 16.
The main pre-processing relates to the interpolation of the figures using the resize method: I tested them searching for the optimal interpolation method to achieve better recognition of eyes and gaze directions. The algorithm used as Interpolator is the Bicubic Interpolation algorithm because although slower than others from a theoretical point of view, it provides better results in this particular experiment. Worth mentioning, however, is the Lanczos Interpolation method, which produces similar results.
Then the images have been converted to grey-scale and re-edited to allow optimal estimation of gaze directions.

### Face and Eyes detection:
To detect the most relevant details in the figures, I use two Feature-based Cascade Classifiers already implemented in the OpenCV library and built with Machine Learning by training large samples of images. It is possible to identify the people's faces and their eyes in the figures with the "detectMultiScale" method. We tried various combinations of parameters to identify the one that allows us to identify more correct elements in the image.

![Immagini2e3](https://user-images.githubusercontent.com/83292347/147979252-ac643182-ec72-40c2-9295-7b1cee26dcfa.png)

Figure 1: Detected faces and eyes of images 3 and 9.

A noteworthy image is figure Figure 2, as there are many faces oriented in various ways and with other obstacles (e.g. glasses).

![Immagine2](https://user-images.githubusercontent.com/83292347/147979381-f6b966fa-6738-4a4e-881e-787f733ad584.png)

Figure 2: Detected faces and eyes of image 2.

In this image some limits of the Feature-based Cascade Classifiers are shown: there is one face on the bottom bigger than it should be and 2 eyes inside two different faces are not recognized. One in the top image corner, probably because the glasses prevent a correct classification and the other one in the center, probably because the head is slightly turned and the eye is not fully displayed.

### Gaze Estimation: 
#### I use the "HoughCircle" method to estimate the gaze direction.

First, the images found from the eyes detection receive processing: they are equalized and converted to black and white. To avoid ruining the size of the iris, I do not use morphological operators. Then I filter the figures to remove some imperfections and improve the HoughCircle algorithm looking for the best combination of parameters.

![IrisImmagine7](https://user-images.githubusercontent.com/83292347/147979481-c5a31e17-2c1a-42dd-9183-f2837c622a12.png)

Figure 3: Binary image of the eye in image 7.

To estimate the Gaze Direction, I use the position of the pupil inside the eye image. When the center of the eye is below the 45% line, the eye is looking to the left, between 45% and 60% looking straight, and above 60% looking right. Example:

![PupillaImmagine3](https://user-images.githubusercontent.com/83292347/147979583-7c87d7fe-ed9f-4f0e-ac69-13ebe3cd45f2.png)

Figure 4: Left eye of image 3 looking right.

![DirezioneSguardoImmagine8](https://user-images.githubusercontent.com/83292347/147979634-5c319ee3-8738-415c-9144-c741cddaccaa.png)

Figure 5: Arrows in the direction of image 8 Gaze Detection.

### Results and Resize method:
The outcome of this approach is heavily influenced by the scaling factor used in the resize method to improve the images with poor quality by increasing the number of pixels. The quality of the figures is critical to have a correct gaze direction, so interpolation is crucial. To get the exact estimate of the gaze of all images with [OpenCV 4.0.1](https://opencv.org/release/opencv-4-0-1/), it is necessary to use a slightly different scale factor for each figure. I choose the one that gives the best results, and in the future, better OpenCV methods will provide better results. With [OpenCV 4.0.1](https://opencv.org/release/opencv-4-0-1/), half of the images are correct while in the other half the direction of gaze is not always optimally estimated. There are no parameters that correctly classify all the eyes: a lot depends on the picture quality.

An alternative method is to have a sufficiently trained Deep Learning algorithm. Without using Deep Learning the pupil can be found and classified through blob detection.

The tests, however, demonstrate the goodness and margin for improvement of feature-based cascade classifiers.
