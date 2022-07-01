#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       
#define IM_TYPE	CV_8UC3
#define L 256
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
float** cal_CDF_RGB(Mat& input) {

	int count[L][3] = { 0 };
	float** CDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		CDF[i] = (float*)calloc(3, sizeof(float));

	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			for (int k = 0; k < 3; k++)
				count[input.at<Vec3b>(i, j)[k]][k]++;

	for (int i = 0; i < L; i++) {
		for (int j = 0; j < 3; j++) {
			CDF[i][j] = (float)count[i][j] / (float)(input.rows * input.cols);

			if (i != 0)
				CDF[i][j] += CDF[i - 1][j];
		}
	}
	return CDF;
}
Mat Laplacianfilter(const Mat input);
void hist_eq_Color(Mat& input, Mat& equalized, G(*trans_func)[3], float** CDF);
int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = Laplacianfilter(input); 
	Mat equalized_RGB = output.clone();
	G trans_func_eq_RGB[L][3] = { 0 };
	float** CDF_RGB = cal_CDF_RGB(output);
	hist_eq_Color(output, equalized_RGB, trans_func_eq_RGB, CDF_RGB);

	hist_eq_Color(output, equalized_RGB, trans_func_eq_RGB, CDF_RGB);
	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", equalized_RGB);


	waitKey(0);

	return 0;
}


Mat Laplacianfilter(const Mat input) {


	int row = input.rows;
	int col = input.cols;
	int tempa;
	int tempb;
	int n = 1; // Laplacian Filter Kernel N

	// Initialiazing Kernel Matrix with 3x3 size 
	//Fill code to initialize Laplacian filter kernel matrix  (Given in the lecture notes)
	char Laplacian[] = { 0,1,0,1,-4,1,0,1,0 };
	Mat kernel(3, 3, CV_8SC1, Laplacian);

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process 
					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum1_r += sqrt((float)(input.at<C>(tempa, tempb)[0]) * 2 * kernel.at<G>(a + n, b + n));
					sum1_g += sqrt((float)(input.at<C>(tempa, tempb)[1]) * 2 * kernel.at<G>(a + n, b + n));
					sum1_b += sqrt((float)(input.at<C>(tempa, tempb)[2]) * 2 * kernel.at<G>(a + n, b + n));
				}

			}
			output.at<C>(i, j)[0] = (G)((sum1_r + sum1_g + sum1_b) / 3);
			output.at<C>(i, j)[1] = (G)((sum1_r + sum1_g + sum1_b) / 3);
			output.at<C>(i, j)[2] = (G)((sum1_r + sum1_g + sum1_b) / 3);
		}
	}

	return output;
}
void hist_eq_Color(Mat& input, Mat& equalized, G(*trans_func)[3], float** CDF) {

	for (int i = 0; i < L; i++) {
		for (int j = 0; j < 3; j++) {
			trans_func[i][j] = (G)((L - 1) * CDF[i][j]);
		}
	}
	// perform the transfer 
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			for (int k = 0; k < 3; k++) {
				equalized.at<Vec3b>(i, j)[k] = trans_func[input.at<Vec3b>(i, j)[k]][k];
			}
		}
	}
}