// OSP_Assignment06_Readme.txt

3. Harris_corner-skeleton.cpp
  : perform Harris Corner Detector. 
  - boundary processing method is fixed to mirroring
  - The input image file "checkerboard2.jpg" should be under the project directory.
  <Important Functions>  
  1) int main
    - variables:    
        input: input image(color) data in matrix form
        row: total number of rows in the input image
        col: total number of cols in the input image
        input _gray: input image(grayscale) data in matrix form
        input_visual: clone input image data in matrix form
        output: output image from Harris corner detector in matrix form
        output_norm: normalized output image in matrix form
        corner_mat: matrix marked 1 with corner and 0 with rest
        NonMaxSupp: true if non-maximum suppression is performed
        Subpixel: true if subpixel refinement is performed
        corner_num:  the number of corners
  2) Mat NonMaximum_Suppression
    : corner_mat = 1 for corner, 0 otherwise.
    - parameters:
        input: input image data in matrix form (grayscale)
        corner_mat: matrix marked 1 with corner and 0 with rest
        radius: integer that determines kernel size (kernel size = 2*radius+1)
    - variables: 
        row: total number of rows in the input image
        col: total number of cols in the input image
        input_mirror: (2n+row)X(2n+col) image data with mirroring techniques