// OSP_Assignment05_Readme.txt

2. kmeansSkeleton.cpp
  : perform image segmentation by using K-means Clustering. 
  - The input image file "lena.jpg" should be under the project directory.
  <Functions>  
  1) int main
    - variables:    
        input: input image(color) data in matrix form
        input _gray: input image(grayscale) data in matrix form
        sigma: ratio between intensity (or color) and position
        sum_r: denominator value used to normalize the row position
        sum_c: denominator value used to normalize the column position
        samples: sample image(for RGB data) data in matrix form 
        samples_Pos: sample image(for RGB+position data) data in matrix form
        samples_Gray: sample image(for Intensity data) data in matrix form
        samples_Gray_Pos: sample image(for Intensity+position data) data in matrix form
        clusterCount: the number of clusters
        labels: cluster number data of each pixel in matrix form
        attempts: the number of times the algorithm is executed using different initial labellings
        centers: center pixel data of each cluster (color image)
        centers_G: center pixel data of each cluster (grayscale image)
        new_image: segmented image(for RGB input) data in matrix form
        new_image_pos: segmented image(for RGB+position input) data in matrix form
        new_image_gray: segmented image(for Intensity input) data in matrix form
        new_image_gray_pos: segmented image(for Intensity+position input) data in matrix form
