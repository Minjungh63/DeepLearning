#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt);
Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

												// Add noise to original image
	Mat noise_Gray = Add_salt_pepper_Noise(input_gray, 0.1f, 0.1f);
	Mat noise_RGB = Add_salt_pepper_Noise(input, 0.1f, 0.1f);

	// Denoise, using median filter
	int window_radius = 2;
	Mat Denoised_Gray = Salt_pepper_noise_removal_Gray(noise_Gray, window_radius, "mirroring"); //  the type of boundary processing: zero-padding, mirroring, adjustkernel
	Mat Denoised_RGB = Salt_pepper_noise_removal_RGB(noise_RGB, window_radius, "mirroring");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Impulse Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Impulse Noise (Grayscale)", noise_Gray);

	namedWindow("Impulse Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Impulse Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	waitKey(0);

	return 0;
}

Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp)
{
	Mat output = input.clone();
	RNG rng; // Random Number Generator
	// ps: density of salt noise (0~1), pp: density of pepper noise (0~1)
	int amount1 = (int)(output.rows * output.cols * pp); // ammount of pepper noise
	int amount2 = (int)(output.rows * output.cols * ps);  // amount of salt noise

	int x, y;

	// Grayscale image
	if (output.channels() == 1) {
		for (int counter = 0; counter < amount1; ++counter) // generate pepper noise
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 0; // rng.uniform(0,range): 범위내에 균등 분포를 따르는 난수 생성

		for (int counter = 0; counter < amount2; ++counter) // generate salt noise
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 255;
	}
	// Color image	
	else if (output.channels() == 3) {
		for (int counter = 0; counter < amount1; ++counter) { // generate pepper noise
			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[0] = 0;

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 0;

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 0;
		}

		for (int counter = 0; counter < amount2; ++counter) { // generate salt noise
			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[0] = 255;

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 255;

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 255;
		}
	}

	return output;
}

Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int median1, median2 ;		// index of median value
	int tempx, tempy;

	// initialize median filter kernel
	Mat kernel = Mat::zeros(kernel_size * kernel_size, 1, input.type());
	
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int count = 0;
			if (!strcmp(opt, "zero-padding")) {
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						
						/* Median filter with "zero-padding" boundary process:		

						Fill the code:
						*/
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							kernel.at<G>(count, 0) = input.at<G>(i + x, j + y);
						}
						count++;
					}
				}
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						/* Median filter with "mirroring" boundary process:

						Fill the code:
						*/
						if (i + x > row - 1) {
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						kernel.at<G>(count, 0) = input.at<G>(tempx, tempy);
						count++;
					}
				}
			}

			else if (!strcmp(opt, "adjustkernel")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						
						/* Median filter with "adjustkernel" boundary process:		

						Fill the code:
						*/
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							kernel.at<G>(count, 0) = input.at<G>(i + x, j + y);
							count++;
						}
					}
				}
			}
			// count는 median 값을 구하기 위해 참조할 픽셀 개수
			Mat kernel2= kernel.rowRange(0, count); // kernel2에 median 값을 구하는 데 참조할 픽셀만 저장
			// Sort the kernels in ascending order
			sort(kernel2, kernel2, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			
			if (count % 2 == 1) {  // count가 홀수면 가운데 수를 (i, j)값으로
				median1 = count / 2 ;
				output.at<G>(i, j) = kernel2.at<G>(median1, 0);
			}
			else { // count가 짝수면 가운데 두 수의 평균을 (i, j)값으로
				median1 = count / 2;
				median2 = count / 2 - 1;
				output.at<G>(i, j) = (G)((float)(kernel2.at<G>(median1, 0) + kernel2.at<G>(median2, 0)) / 2);
			}
		}
	}

	return output;
}

Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int median1 = -1, median2 = -1;		// index of median value
	int tempx, tempy;
	int channel = input.channels();

	// initialize ( (TypeX with 3 channel) - (TypeX with 1 channel) = 16 )
	// ex) CV_8UC3 - CV_8U = 16
	Mat kernel = Mat::zeros(kernel_size * kernel_size, channel, input.type() - 16);
	
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int count = 0;
			if (!strcmp(opt, "zero-padding")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						
						/* Median filter with "zero-padding" boundary process:		

						Fill the code:
						*/
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							kernel.at<C>(count, 0)[0] = input.at<C>(i + x, j + y)[0];
							kernel.at<C>(count, 0)[1] = input.at<C>(i + x, j + y)[1];
							kernel.at<C>(count, 0)[2] = input.at<C>(i + x, j + y)[2];
						}
						count++;
					}
				}
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						/* Median filter with "mirroring" boundary process:

						Fill the code:
						*/
						if (i + x > row - 1) {
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						kernel.at<C>(count, 0)[0] = input.at<C>(tempx, tempy)[0];
						kernel.at<C>(count, 0)[1] = input.at<C>(tempx, tempy)[1];
						kernel.at<C>(count, 0)[2] = input.at<C>(tempx, tempy)[2];
						count++;
					}
				}
			}

			else if (!strcmp(opt, "adjustkernel")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						
						/* Median filter with "adjustkernel" boundary process:		

						Fill the code:
						*/
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							kernel.at<C>(count, 0)[0] = input.at<C>(i + x, j + y)[0];
							kernel.at<C>(count, 0)[1] = input.at<C>(i + x, j + y)[1];
							kernel.at<C>(count, 0)[2] = input.at<C>(i + x, j + y)[2];
							count++;
						}
					}
				}
			}
			// count는 median 값을 구하기 위해 참조할 픽셀 개수
			Mat kernel2 = kernel.rowRange(0, count); // kernel2에 median 값을 구하는 데 참조할 픽셀만 저장
			// Sort the kernels in ascending order
			sort(kernel2, kernel2, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			if (count % 2 == 1) { // count가 홀수면 가운데 수를 (i, j)값으로
				median1 = count / 2;
				output.at<C>(i, j)[0] = kernel2.at<C>(median1, 0)[0];
				output.at<C>(i, j)[1] = kernel2.at<C>(median1, 0)[1];
				output.at<C>(i, j)[2] = kernel2.at<C>(median1, 0)[2];
			}
			else { // count가 짝수면 가운데 두 수의 평균을 (i, j)값으로
				median1 = count / 2;
				median2 = count / 2 - 1;
				output.at<C>(i, j)[0] = (G)((float)(kernel2.at<C>(median1, 0)[0] + kernel2.at<C>(median2, 0)[0]) / 2);
				output.at<C>(i, j)[1] = (G)((float)(kernel2.at<C>(median1, 0)[1] + kernel2.at<C>(median2, 0)[1]) / 2);
				output.at<C>(i, j)[2] = (G)((float)(kernel2.at<C>(median1, 0)[2] + kernel2.at<C>(median2, 0)[2]) / 2);
			}
		}
	}

	return output;
}
