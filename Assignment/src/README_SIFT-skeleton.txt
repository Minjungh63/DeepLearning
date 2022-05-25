Assignment07_Readme_2071044정민정
(1) SIFT-skeleton.cpp
-purpose: executes the SIFT descriptor and then implements feature matching.
-how to run this code: 
	1. prepare two images to match.
	2. decide whether or not to apply the crossCheck condition and ratio_threshold condition.
	3. wait untill all the results come out.
-important functions:
	1. euclidDistance: Calculate euclid distance
	2. nearestNeighbor: Find the index of nearest neighbor point from keypoints
	3. secondNearestNeighbor: Find the index of second nearest neighbor point from keypoints.
	4. findPairs: Find pairs of points with the smallest distace between them.