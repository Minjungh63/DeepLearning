// OSP_Assignment05_Readme.txt

1. adaptivethresholdSkeleton.cpp
  : perform image segmentation by using adaptive thresholding. 
  - boundary processing method is fixed to zero-padding
  - The input image file "writing.jpg" should be under the project directory.
  <Functions>  
  1) int main
    - variables:    
        input: input image(color) data in matrix form
        input _gray: input image(grayscale) data in matrix form
        output: output image(segmented image) data in matrix form
  2) Mat adaptive_thres
    : perform image segmentation by using adaptive thresholding.
    - parameters:
        input: input image data in matrix form
        n: integer that determines kernel size (kernel size= 2*n+1)
        bnumber: weight value
    - variables: 
        row: total number of rows in the input image
        col: total number of cols in the input image
        kernel_size: the number of kernel size
        kernelvalue = value of kernel matrix
        output: segmented image data in matrix form