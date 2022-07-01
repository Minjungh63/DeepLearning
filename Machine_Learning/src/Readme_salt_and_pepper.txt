// OSP_Assignment04_Readme.txt

1. salt_and_pepper.cpp
  : generate salt and pepper noise and remove it by using medianfilter methods. 
  - user can choose boundary processing method: zero-padding, mirroring, adjusting the filter kernel
  - The input image file "lena.jpg" should be under the project directory.
  <Functions>  
  1) int main
    - variables:    
        input: input image(color) data in matrix form
        input _gray: input image(grayscale) data in matrix form
        noise_Gray: input image(grayscale) + salt and pepper noise data in matrix form
        noise_RGB: input image(color) + salt and pepper noise data in matrix form
        window_radius: integer that determines kernel size (kernel size= 2*window_radius+1) 
        Denoised_Gray: denoised image(grayscale) data in matrix form
        Denoised_RGB: denoised image(color) data in matrix form
  2) Mat Add_salt_pepper_Noise
    : generate salt and pepper noise proportional to the given ps, pp. Then add it to the input image.
    - parameters:
        input: input image data in matrix form
        ps: density of salt noise(0~1)
        pp: density of pepper noise(0~1)
    - variables: 
        output: image data with salt and pepper noise in matrix form
        rng: reference of Random Number Generator
        amount1: the number of pepper noises
        amount2: the number of salt noise
        x: randomly generated row number
        y: randomly generated col number
  3) Mat Salt_pepper_noise_removal_Gray
    : remove salt and pepper noise from grayscale image.
    - parameters:
       input: input image data in matrix form
       n: nteger that determines kernel size (kernel size= 2*n+1)
       opt: the type of boundary processing
    - variables:
       row: total number of rows in the input image
       col: total number of cols in the input image
       kernel_size: the number of kernel size
       median1: index number of the median value
       median2: Index number of the median value (if there are two median values, use this variable)
       tempx: Temporary x-value (use in mirroring method)
       tempy: Temporary y-value (use in mirroring method)
       kernel: kernel data in matrix form
       output: denoised image(grayscale) data in matrix form
  4) Mat Salt_pepper_noise_removal_RGB
    : remove salt and pepper noise from color image.
    - parameters:
       input: input image data in matrix form
       n: nteger that determines kernel size (kernel size= 2*n+1)
       opt: the type of boundary processing
    - variables:
       row: total number of rows in the input image
       col: total number of cols in the input image
       kernel_size: the number of kernel size
       median1: index number of the median value
       median2: Index number of the median value (if there are two median values, use this variable)
       tempx: Temporary x-value (use in mirroring method)
       tempy: Temporary y-value (use in mirroring method)
       channel: total number of channels in the input image
       kernel: kernel data in matrix form
       output: denoised image(color) data in matrix form
