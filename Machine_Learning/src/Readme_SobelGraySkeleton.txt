#purpose
Apply Sobel Filter to grayscale image
# how to run this code
Input the image path into the parameter of "imread" function. 
Then sobelfilter function will apply sobel filter to image. (Sx={-1,0,1,-2,0,2,-1,0,1}, Sy={-1,-2,1,0,0,0,1,2,1})
This code uses "mirroring" boundary processing method.
After getting all the values of output, apply histogram equalization to output. This will visualize the output by mapping the output value into [0~255].
# how to define default parameter
1. UnsharpMash function: input image(grayscale image)
2. hist_eq: input image, equalized image(->save output image), trans_func(-> save trans_func value),CDF of input image
