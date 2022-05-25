#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // RGB input image
	Mat input_gray; // Grayscale input image

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}
	cvtColor(input, input_gray, COLOR_RGB2GRAY);

	namedWindow("Original(RGB)", WINDOW_AUTOSIZE);
	imshow("Original(RGB)", input);
	namedWindow("Original(Grayscale)", WINDOW_AUTOSIZE);
	imshow("Original(Grayscale)", input_gray);

	float sigma = 1; // sigma: adjusting the ratio between intensity (or color) and position
	float sum_r = 0.0, sum_c = 0.0; // use to normalize position
	Mat samples(input.rows * input.cols, 3, CV_32F); // (R, G, B)
	Mat samples_Pos(input.rows * input.cols, 5, CV_32F); // (R, G, B, row/sigma, col/sigma)
	Mat samples_Gray(input_gray.rows * input_gray.cols, 1, CV_32F); // (I)
	Mat samples_Gray_Pos(input_gray.rows * input_gray.cols, 3, CV_32F); // (I, row/sigma, col/sigma)
	for (int y = 0; y < input.rows; y++) {
		sum_r += (float)(y / sigma); 
		for (int x = 0; x < input.cols; x++) {
			sum_c += (float)(x / sigma);
			samples_Gray.at<float>(y + x * input.rows, 0) = (float)(input_gray.at<uchar>(y, x) / 255.0);
			samples_Gray_Pos.at<float>(y + x * input.rows, 0) = (float)(input_gray.at<uchar>(y, x) / 255.0);
			for (int z = 0; z < 3; z++) {
				samples.at<float>(y + x * input.rows, z) = (float)(input.at<Vec3b>(y, x)[z] / 255.0);
				samples_Pos.at<float>(y + x * input.rows, z) = (float)(input.at<Vec3b>(y, x)[z] / 255.0);
			}
		}
	}
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			samples_Gray_Pos.at<float>(y + x * input.rows, 1) = (float)(y / sigma) / sum_r;
			samples_Gray_Pos.at<float>(y + x * input.rows, 2) = (float)(x / sigma) / sum_c;
			samples_Pos.at<float>(y + x * input.rows, 3) = (float)(y / sigma) / sum_r;
			samples_Pos.at<float>(y + x * input.rows, 4) = (float)(x / sigma) / sum_c;
		}
	}

	// Clustering is performed for each channel
	// Note that the intensity value is not normalized here (0~1). You should normalize both intensity and position when using them simultaneously.
	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers, centers_G;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	Mat new_image(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			new_image.at<Vec3b>(y, x)[0] = (uchar)(centers.at<float>(cluster_idx, 0) * 255);
			new_image.at<Vec3b>(y, x)[1] = (uchar)(centers.at<float>(cluster_idx, 1) * 255);
			new_image.at<Vec3b>(y, x)[2] = (uchar)(centers.at<float>(cluster_idx, 2) * 255);
		}

	namedWindow("clustered image(RGB)", WINDOW_AUTOSIZE);
	imshow("clustered image(RGB)", new_image);

	kmeans(samples_Pos, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	Mat new_image_pos(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			new_image_pos.at<Vec3b>(y, x)[0] = (uchar)(centers.at<float>(cluster_idx, 0) * 255);
			new_image_pos.at<Vec3b>(y, x)[1] = (uchar)(centers.at<float>(cluster_idx, 1) * 255);
			new_image_pos.at<Vec3b>(y, x)[2] = (uchar)(centers.at<float>(cluster_idx, 2) * 255);
		}
	namedWindow("clustered image(RGB and Position)", WINDOW_AUTOSIZE);
	imshow("clustered image(RGB and Position)", new_image_pos);

	kmeans(samples_Gray, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers_G);
	Mat new_image_gray(input_gray.size(), input_gray.type());
	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			new_image_gray.at<uchar>(y, x) = (uchar)(centers_G.at<float>(cluster_idx, 0) * 255);
		}
	namedWindow("clustered image(Grayscale)", WINDOW_AUTOSIZE);
	imshow("clustered image(Grayscale)", new_image_gray);

	kmeans(samples_Gray_Pos, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers_G);
	Mat new_image_gray_pos(input_gray.size(), input_gray.type());
	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			new_image_gray_pos.at<uchar>(y, x) = (uchar)(centers_G.at<float>(cluster_idx, 0) * 255);
		}
	namedWindow("clustered image(Grayscale and Position)", WINDOW_AUTOSIZE);
	imshow("clustered image(Grayscale and Position)", new_image_gray_pos);

	waitKey(0);

	return 0;
}