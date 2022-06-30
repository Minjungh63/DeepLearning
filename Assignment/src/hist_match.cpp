#include "hist_func.h"

void hist_match(Mat& input, Mat& matched, G* trans_func, float* CDF, float* CDF_z);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	const Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray, ref_gray;

	// input, reference : RGB -> Gray scale
	cvtColor(input, input_gray, CV_RGB2GRAY);
	cvtColor(reference, ref_gray, CV_RGB2GRAY);

	Mat matched = input_gray.clone();

	// PDF or transfer function txt files
	FILE* f_matched_PDF, * f_PDF;
	FILE* f_trans_func_match;

	float* PDF = cal_PDF(input_gray);                          // PDF of Input image(grayscale)
	float* CDF = cal_CDF(input_gray);                          // CDF of Input image(grayscale)                   
	float* CDF_ref = cal_CDF(ref_gray);                        // CDF of Reference image(grayscale)

	fopen_s(&f_PDF, "PDF.txt", "w+");
	fopen_s(&f_matched_PDF, "matched_PDF.txt", "w+");
	fopen_s(&f_trans_func_match, "trans_func_match.txt", "w+");

	G trans_func_match[L] = { 0 };                         // transfer function G^(-1)T

	// histogram matching
	hist_match(input_gray, matched, trans_func_match, CDF, CDF_ref);

	// matched PDF
	float* matched_PDF_gray = cal_PDF(matched);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_matched_PDF, "%d\t%f\n", i, matched_PDF_gray[i]);
		// write transfer functions
		// ...
		fprintf(f_trans_func_match, "%d\t%d\n", i, trans_func_match[i]);

	}

	// memory release
	free(PDF);
	free(CDF);
	free(CDF_ref);
	fclose(f_PDF);
	fclose(f_matched_PDF);
	fclose(f_trans_func_match);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matched);

	//////////////////////////////////////////////////////////////
	waitKey(0);

	return 0;
}

// histogram matching
void hist_match(Mat& input, Mat& matched, G* trans_func, float* CDF, float* CDF_z) {

	G trans_func_T[L] = { 0 };                                 // transfer function T (s = T(r))
	G trans_func_G[L] = { 0 };                                 // transfer function G (z = G(z))

	// compute trasfer function T and G
	for (int i = 0; i < L; i++) {
		trans_func_T[i] = (G)((L - 1) * CDF[i]);
		trans_func_G[i] = (G)((L - 1) * CDF_z[i]);
	}
	// compute transfer function (z = G^(-1)T(r))
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < L; j++) {
			if (trans_func_G[j] == trans_func_T[i]) {
				trans_func[i] = j;
				break;
			}
			else if (trans_func_G[j] > trans_func_T[i]) {
				trans_func[i] = trans_func[i - 1] + 1 < j - 1 ? trans_func[i - 1] + 1 : j - 1;
				break;
			}
		}
	}

	// perform the transfer
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			matched.at<G>(i, j) = trans_func[input.at<G>(i, j)];
		}


	}
}
