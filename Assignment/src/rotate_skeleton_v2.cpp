#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;
	
	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "bilinear");

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);

	return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	// rotated image size: sq_row X sq_col
	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	// Initialize to zero
	Mat output = Mat::zeros(sq_row, sq_col, input.type());

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			// Rotate around the center of the image
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			float lamda_x = x - floor(x);
			float lamda_y = y - floor(y);
			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				if (!strcmp(opt, "nearest")) {
					// Choose the nearest point to (y,x)
					float X = lamda_x < 0.5 ? floor(x) : ceil(x);
					float Y = lamda_y < 0.5 ? floor(y) : ceil(y);
					output.at<T>(i, j) = input.at<T>(Y, X);
				}
				else if (!strcmp(opt, "bilinear")) {
					// values for four adjacent points
					T p1 = input.at<T>(floor(y),floor(x));
					T p2 = input.at<T>(ceil(y), floor(x));
					T p3 = input.at<T>(floor(y), ceil(x));
					T p4 = input.at<T>(ceil(y), ceil(x));

					// Applying linear interpolation to p1 and p2
					T P1 = lamda_y * p1 + (1 - lamda_y) * p2;
					// Applying linear interpolation to p3 and p4
					T P2 = lamda_y * p3 + (1 - lamda_y) * p4;
					// Applying linear interpolation to P1 and P2
					output.at<T>(i, j) = lamda_x * P1 + (1 - lamda_x) * P2;
				}
			}
		}
	}

	return output;
}