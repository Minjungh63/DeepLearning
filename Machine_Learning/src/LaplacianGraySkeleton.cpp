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
float* cal_CDF(Mat& input) {

	int count[L] = { 0 };
	float* CDF = (float*)calloc(L, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// Compute CDF
	for (int i = 0; i < L; i++) {
		CDF[i] = (float)count[i] / (float)(input.rows * input.cols);

		if (i != 0)
			CDF[i] += CDF[i - 1];
	}

	return CDF;
}
Mat Laplacianfilter(const Mat input);
void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);
int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;
	
	cvtColor(input, input_gray, CV_RGB2GRAY); // Converting image to gray

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("GrayScale", WINDOW_AUTOSIZE);
	imshow("GrayScale", input_gray);
	output = Laplacianfilter(input_gray); 
	Mat equalized = output.clone();
	G trans_func_eq[L] = { 0 };
	float* CDF = cal_CDF(output);

	hist_eq(output, equalized, trans_func_eq, CDF);
	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", equalized);


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
	Mat kernel(3, 3, CV_8SC1,Laplacian);

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
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
					sum1 += sqrt((float)(input.at<G>(tempa, tempb)) *2*kernel.at<G>(a + n, b + n));
				}

			}
			output.at<G>(i, j)= (G)sum1;
		}
	}
	
	return output;
}
void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}