// OSP_Assignment04_Readme.txt

2. Gaussian.cpp
  : generate gaussian noise and remove it by using gaussianfilter methods. 
  - user can choose boundary processing method: zero-padding, mirroring, adjusting the filter kernel
  - The input image file "lena.jpg" should be under the project directory.
  <Functions>  
  1) int main
    - variables:    
        input: input image(color) data in matrix form
        input _gray: input image(grayscale) data in matrix form
        noise_Gray: input image(grayscale) + gaussian noise data in matrix form
        noise_RGB: input image(color) + gaussian noise data in matrix form
        Denoised_Gray: denoised image(grayscale) data in matrix form
        Denoised_RGB: denoised image(color) data in matrix form
  2) Mat Add_Gaussian_noise
    : generate gaussian noise following a normal distribution of the given mean and standard deviation. Then add it to the input image.
    - parameters:
        input: input image data in matrix form
        mean: mean value of normal distribution
        sigma: standard deviation of normal distribution
    - variables: 
        NoiseArr: image data with gaussian noise in matrix form
        rng: reference of Random Number Generator
  3) Mat Gaussianfilter_Gray
    : remove gaussian noise from grayscale image.
    - parameters:
       input: input image data in matrix form
       n: integer that determines kernel size (kernel size= 2*n+1)
       sigma_t: standard deviation for x-coordinate
       sigma_s: standard deviation for y-coordinate
       opt: the type of boundary processing
    - variables:
       row: total number of rows in the input image
       col: total number of cols in the input image
       kernel_size: the number of kernel size
       kernel: kernel data in matrix form
       denom: denominator value for normalization
       output: denoised image(grayscale) data in matrix form
  4) Mat Gaussianfilter_RGB
    : remove gaussian noise from color image.
    - parameters:
       input: input image data in matrix form
       n: integer that determines kernel size (kernel size= 2*n+1)
       sigma_t: standard deviation for x-coordinate
       sigma_s: standard deviation for y-coordinate
       opt: the type of boundary processing
    - variables:
       row: total number of rows in the input image
       col: total number of cols in the input image
       kernel_size: the number of kernel size
       kernel: kernel data in matrix form
       denom: denominator value for normalization
       output: denoised image(color) data in matrix form
