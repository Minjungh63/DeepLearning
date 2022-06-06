#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

using namespace cv;


Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using bilateral filter
	Mat Denoised_Gray = Bilateralfilter_Gray(noise_Gray, 3, 10, 10, 10, "adjustkernel");
	Mat Denoised_RGB = Bilateralfilter_RGB(noise_RGB, 3, 10, 10, 10, "adjustkernel");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt){

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);

	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// value1: spatial distance, value2: intensity distance
			double denom = 0.0, value1, value2;
			if (!strcmp(opt, "zero-padding")) {
				double sum1 = 0.0;
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
						// (i+x, j+y)�� boundary ���� �ȼ��̸� ���� input.at<double>(i+x,j+y)��
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							value2 = exp(-(pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2)) / (2 * pow(sigma_r, 2)));
						}
						// (i+x, j+y)�� boundary �ܺ� �ȼ��̸� ���� 0����
						else {
							value2 = exp(-(pow(input.at<double>(i, j), 2)) / (2 * pow(sigma_r, 2)));
						}
						kernel.at<double>(x + n, y + n) = value1 * value2;
						denom += value1 * value2;
					}
				}
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						kernel.at<double>(x + n, y + n) /= denom;
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sum1 += kernel.at<double>(x + n, y + n) * input.at<double>(i + x, j + y);
						}
					}
				}
				output.at<double>(i, j) = (double)sum1;

			}

			else if (!strcmp(opt, "mirroring")) {
				double sum1 = 0.0;
				int tempx, tempy;

				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
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
						value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
						value2 = exp(-(pow(input.at<double>(i, j) - input.at<double>(tempx, tempy), 2)) / (2 * pow(sigma_r, 2)));
						kernel.at<double>(x + n, y + n) = value1 * value2;
						denom += value1 * value2;
					}
				}
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
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
						kernel.at<double>(x + n, y + n) /= denom;
						sum1 += kernel.at<double>(x + n, y + n) * input.at<double>(tempx, tempy);
					}
				}
				output.at<double>(i, j) = (double)sum1;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				double sum1 = 0.0;
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							value2 = exp(-(pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2)) / (2 * pow(sigma_r, 2)));
							kernel.at<double>(x + n, y + n) = value1 * value2;
							denom += value1 * value2;
						}
					}
				}
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							kernel.at<double>(x + n, y + n) /= denom;
							sum1 += kernel.at<double>(x + n, y + n) * input.at<double>(i + x, j + y);
						}
					}
				}
				output.at<double>(i, j) = (double)sum1;

			}

		}
	}
	return output;
}

Mat  Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);

	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			double denom = 0.0, value1, value2;
			if (!strcmp(opt, "zero-padding")) {
				double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							value2 = exp(-(pow(input.at<Vec3d>(i, j)[0] - input.at<Vec3d>(i + x, j + y)[0], 2)
								+ pow(input.at<Vec3d>(i, j)[1] - input.at<Vec3d>(i + x, j + y)[1], 2)
								+ pow(input.at<Vec3d>(i, j)[2] - input.at<Vec3d>(i + x, j + y)[2], 2)) / (2 * pow(sigma_r, 2)));
						}
						else {
							value2 = exp(-(pow(input.at<Vec3d>(i, j)[0], 2)
								+ pow(input.at<Vec3d>(i, j)[1], 2)
								+ pow(input.at<Vec3d>(i, j)[2], 2)) / (2 * pow(sigma_r, 2)));
						}
						kernel.at<double>(x + n, y + n) = value1 * value2;
						denom += value1 * value2;
					}
				}
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						kernel.at<double>(x + n, y + n) /= denom;
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sum1 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[0];
							sum2 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[1];
							sum3 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[2];
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)sum1;
				output.at<Vec3d>(i, j)[1] = (double)sum2;
				output.at<Vec3d>(i, j)[2] = (double)sum3;

			}

			else if (!strcmp(opt, "mirroring")) {
				double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
				int tempx, tempy;

				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
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
						value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
						value2 = exp(-(pow(input.at<Vec3d>(i, j)[0] - input.at<Vec3d>(tempx, tempy)[0], 2)
							+ pow(input.at<Vec3d>(i, j)[1] - input.at<Vec3d>(tempx, tempy)[1], 2)
							+ pow(input.at<Vec3d>(i, j)[2] - input.at<Vec3d>(tempx, tempy)[2], 2)) / (2 * pow(sigma_r, 2)));
						kernel.at<double>(x + n, y + n) = value1 * value2;
						denom += value1 * value2;
					}
				}
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
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
						kernel.at<double>(x + n, y + n) /= denom;
						sum1 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(tempx, tempy)[0];
						sum2 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(tempx, tempy)[1];
						sum3 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(tempx, tempy)[2];
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)sum1;
				output.at<Vec3d>(i, j)[1] = (double)sum2;
				output.at<Vec3d>(i, j)[2] = (double)sum3;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							value2 = exp(-(pow(input.at<Vec3d>(i, j)[0] - input.at<Vec3d>(i + x, j + y)[0], 2)
								+ pow(input.at<Vec3d>(i, j)[1] - input.at<Vec3d>(i + x, j + y)[1], 2)
								+ pow(input.at<Vec3d>(i, j)[2] - input.at<Vec3d>(i + x, j + y)[2], 2)) / (2 * pow(sigma_r, 2)));
							kernel.at<double>(x + n, y + n) = value1 * value2;
							denom += value1 * value2;
						}
					}
				}
				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							kernel.at<double>(x + n, y + n) /= denom;
							sum1 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[0];
							sum2 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[1];
							sum3 += kernel.at<double>(x + n, y + n) * input.at<Vec3d>(i + x, j + y)[2];
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)sum1;
				output.at<Vec3d>(i, j)[1] = (double)sum2;
				output.at<Vec3d>(i, j)[2] = (double)sum3;

			}

		}
	}

	return output;
}