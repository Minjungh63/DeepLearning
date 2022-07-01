# hist_eq readme file
# Purpose of this code
This code performs histogram equalization of the image.
# How to run this code
Input the image path into the parameter of "imread" function. 
Then this code will process the image as follows.
1. "cvtColor" function convert the image to grayscale.
2. Create(or choose) text files f_PDF to save PDF data of the input image.
   Create(or choose) text files f_equalized_PDF_gray to save PDF data of the equalized image.
   Create(or choose) text files f_trans_func_eq to save transfer function data.
3. Use "hist_eq" function to compute transfer function and get the equalized histogram data.
4. Display the input image and the equalized image on gray scale in the window.
# How to adjust parameters
1. You can change the input image by adjusting the first parameter of the imread function.
If write the path of desired input image in the first parameter of the imread function, you can change an input image.
# How to define default parameters
"hist_eq" function has 4 parameters. 
Input image data, variable for saving equalized image data, variable for saving results of transfer function, and CDF data of the input image.