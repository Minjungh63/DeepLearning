# hist_eq_YUV readme file
# Purpose of this code
This code performs histogram equalization of the color image YUV.
# How to run this code
Input the image path into the parameter of "imread" function. 
Then this code will process the image as follows.
1. Create(or choose) text files f_PDF_RGB to save PDF data of the color input image.
   Create(or choose) text files f_equalized_PDF_YUV to save PDF data of the color equalized image.
   Create(or choose) text files f_trans_func_eq_YUV to save transfer function data.
2. Convert the input image to YUV and divide each channel into channel [0], channel [1], and channel [2].
3. Use "cal_PDF_RGB" and "cal_CDF" function to generate PDF of the input image and CDF of the Y channel of input image.
4. Use "hist_eq" function to compute transfer function and get the equalized histogram data of Y channel.
5. Use "merge" function to merge equalized Y, U of input image, and V of input image.
6. Convert the merged image to RGB.
7. Display the input image and the equalized image on color-mode in the window.
# How to adjust parameters
1. You can change the input image by adjusting the first parameter of the imread function.
If write the path of desired input image in the first parameter of the imread function, you can change an input image.
# How to define default parameters
1. "hist_eq" function has 4 parameters. Input image data, variable for saving equalized image data, variable for saving results of transfer function, and CDF data of the Y channel of input image.
2. "cal_PDF_RGB" / "cal_CDF" function
After using this function, the memory must be released by the **free** function.
