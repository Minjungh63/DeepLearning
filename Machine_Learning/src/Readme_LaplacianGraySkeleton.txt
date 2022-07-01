#purpose
Apply Laplacian Filter to grayscale image
# how to run this code
Input the image path into the parameter of "imread" function. 
Then Laplacianfilter function will apply laplacian filter to image.(Laplacian= { 0,1,0,1,-4,1,0,1,0 })
This code uses "mirroring" boundary processing method.
After getting all the values of output, apply histogram equalization to output. This will visualize the output by mapping the output value into [0~255].
# how to define default parameter
1. Laplacianfilter function: input image(grayscale)
2. hist_eq: input image, equalized image(->save output image), trans_func(-> save trans_func value),CDF of input image