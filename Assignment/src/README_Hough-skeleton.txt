Hough-skeleton.cpp README 
2071044 정민정

1. purpose: estimate the line segments by using Hough transform.
2. How to run this code
	(1) Run the canny edge detector
	(2) Run the Hough transform by using HoughLines or HoughLinesP
	(3) Draw the line fitting result
3. important function
	(1) HoughLines: standard Hough transform. using rho, theta, and threshold.
	(2) HoughLinesP: probabilistic Hough transform. using threshold, min_line_length, and max_line_gap.
	(3) Canny: run the canny edge detector.