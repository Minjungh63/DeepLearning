# hist_eq_RGB readme file
# Purpose of this code
This code performs histogram equalization of the color image.
# How to run this code
Input the image path into the parameter of "imread" function. 
Then this code will process the image as follows.
1. Create(or choose) text files f_PDF_RGB to save PDF data of the color input image.
   Create(or choose) text files f_equalized_PDF_RGB to save PDF data of the color equalized image.
   Create(or choose) text files f_trans_func_eq_RGB to save transfer function data.
2. Use "cal_PDF_RGB" and "cal_CDF_RGB" function to generage PDF and CDF of the input image.
3. Use "hist_eq_Color" function to compute transfer function and get the equalized histogram data.
4. Display the input image and the equalized image on color-mode in the window.
# How to adjust parameters
1. You can change the input image by adjusting the first parameter of the imread function.
If write the path of desired input image in the first parameter of the imread function, you can change an input image.
# How to define default parameters
1. "hist_eq_Color" function has 4 parameters. Input image data, variable for saving equalized image data, variable for saving results of transfer function, and CDF data of the input image.
2. "cal_PDF_RGB" / "cal_CDF_RGB" function has one parameter, input image.
Return type of "cal_PDF_RGB" / "cal_CDF_RGB" function is "float **", parameter should be a color image.
After using this function, the memory must be released by the **free** function.
