#purpose
Apply Separable Gaussian Filter to grayscale image
# how to run this code
Input the image path into the parameter of "imread" function. 
User can change boudary processing method and sigmaS, sigmaT value (Gaussianfilter_sep function).
Then Gaussianfilter_sep function will apply gaussian filter to image.
# how to define default parameter
1.  Gaussianfilter_sep function: input image, n(->kernelsize:2n+1),sigmaT,sigmas,boundary processing method