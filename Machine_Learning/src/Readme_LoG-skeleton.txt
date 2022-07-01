// OSP_Assignment06_Readme.txt

1. LoG-skeleton.cpp
  : perform Laplacian of Gaussian. 
  - boundary processing method is fixed to mirroring
  - The input image file "lena.jpg" should be under the project directory.
  <Important Functions>  
  1) int main
    - variables:    
        input: input image(color) data in matrix form
        input _gray: input image(grayscale) data in matrix form
        window_radius: integer that determines kernel size (kernel size = 2*window_radius+1)
        sigma_t: sigma value of gaussian filter
        sigma_s: sigma value of gaussian filter
        h_f: result image data from gaussian filtering (grayscale)
        h_f_RGB: result image data from gaussian filtering (color)
        Laplacian: result image data from laplacian filtering (grayscale)
        Laplacian_RGB: result image data from laplacian filtering (color)
  2) Mat Gaussianfilter
    : perform Gaussian filtering (grayscale image).
    - parameters:
        input: input image data in matrix form (grayscale)
        n: integer that determines kernel size (kernel size= 2*n+1)
        sigma_t: sigma value of gaussian filter
        sigma_s: sigma value of gaussian filter
    - variables: 
        row: total number of rows in the input image
        col: total number of cols in the input image
        kernel: kernel data of gaussian filter
        output: output image from gaussian filtering in matrix form
        kernelvalue = value of kernel matrix
        input_mirror: (2n+row)X(2n+col) image data with mirroring techniques
  3) Mat Gaussianfilter_RGB
    : perform Gaussian filtering (color image).
    - parameters:
        input: input image data in matrix form (color)
        n: integer that determines kernel size (kernel size= 2*n+1)
        sigma_t: sigma value of gaussian filter
        sigma_s: sigma value of gaussian filter
    - variables: 
        row: total number of rows in the input image
        col: total number of cols in the input image
        kernel: kernel data of gaussian filter
        output: output image from gaussian filtering in matrix form
        kernelvalue = value of kernel matrix
        input_mirror: (2n+row)X(2n+col) image data with mirroring techniques
  4) Mat Laplacianfilter
    : perform Laplacian filtering.
    - parameters:
        input: input image data in matrix form (grayscale)
    - variables: 
        row: total number of rows in the input image
        col: total number of cols in the input image
        kernel: kernel data of gaussian filter
        output: output image from gaussian filtering in matrix form
        input_mirror: (2n+row)X(2n+col) image data with mirroring techniques
  5) Mat Laplacianfilter_RGB
    : perform Laplacian filtering (color).
    - parameters:
        input: input image data in matrix form (color)
    - variables: 
        row: total number of rows in the input image
        col: total number of cols in the input image
        kernel: kernel data of gaussian filter
        output: output image from gaussian filtering in matrix form
        input_mirror: (2n+row)X(2n+col) image data with mirroring techniques