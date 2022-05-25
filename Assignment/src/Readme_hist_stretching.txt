# hist_stretching readme file
# Purpose of this code
This code stretches the histogram data to enhance contrast of the image.
# How to run this code
Input the image path into the parameter of "imread" function. 
Next, input the (x1,y1), (x2,y2) values into the parameter of "linear_stretching" function.
Then this code will process the image as follows.
1. "cvtColor" function convert the image to grayscale.
2. Create(or choose) text files f_PDF to save PDF data of the input image.
   Create(or choose) text files f_stretched_PDF to save PDF data of the stretched image.
   Create(or choose) text files f_trans_func_stretch to save transfer function data.
3. Use "linear_stretching" function to compute transfer function and get the stretched histogram data.
4. Display the input image and the stretched image on gray scale in the window.
# How to adjust parameters
1. You can change the input image by adjusting the first parameter of the imread function.
If write the path of desired input image in the first parameter of the imread function, you can change an input image.
2. You can choose different (x1, y1), (x2, y2)
If write another x1, x2, y1, y2 in fourth~seventh parameters of the linear_stretching function, you can choose different linear stretching function.
# How to define default parameters
"linear_stretching" function has 7 parameters. 
Input image data, variable for saving stretched image data, variable for saving results of transfer function, and 4 values of input image x1,x2,y1,y2.