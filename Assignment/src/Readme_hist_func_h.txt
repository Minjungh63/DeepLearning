# hist_func.h readme file
# Purpose of this code
This header file provides a function for generating PDF and CDF of image (grayscale and color) and a image type (use typedef keyword).
# How to define default parameters
1. cal_PDF / cal_CDF function 
This function generates PDF/CDF data of the image which passed as a parameter.
In this code, return type of cal_PDF / cal_CDF function is "float *", so parameter should be a grayscale image.
After using this function, the memory must be released by the **free** function.
2. cal_PDF_RGB / cal_CDF_RGB function
This function generates PDF/CDF data of the image which passed as a parameter.
In this code, return type of cal_PDF / cal_CDF function is "float **", so parameter should be a color image.
After using this function, the memory must be released by the **free** function.