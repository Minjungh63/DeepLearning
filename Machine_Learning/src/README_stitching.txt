stitching.cpp README
2071044 정민정

1. purpose: run SIFT descriptor to implement feature matching. Then, calculate affine transformation and print image stitching result.
2. how to run this code:
	(1) Feature matching
	: Run SIFT descriptor for two images, and perform the feature matching using NN. NExt, refine the feature matching results using both cross-checking and ratio-based thresholding.
	(2) Affine transform estimation
	: implement transform estimation by two way. (Mx=b and Mx=b + RANSAC)
	(3) Perform image stitching
3. important functions:
	(1) cal_affine: calcutate affine transformation.
	(2) blend_stitching: stitch two images
	(3) findPairs: implement feature matching.