# PDF_CDF readme file
# Purpose of this code
This code generates PDF and CDF of the input image(grayscale) as txt files.
# How to run this code
Input the image path into the parameter of "imread" function. Then this code will process the image as follows.
1. "cvtColor" function convert the image to grayscale.
2. Create(or choose) text files f_PDF and f_CDF to save PDF and CDF data of the image.
3. Use "cal_PDF" and "cal_CDF" functions to generate PDF and CDF data  of the image.
4. Use "fprintf" function to write PDF and CDF data on PDF.txt and CDF.txt.
5. Display the image in gray scale and color mode on the window.
# How to adjust parameters (if any)
1. You can change the input image by adjusting the first parameter of the imread function.
If write the path of desired input image in the first parameter of the imread function, you can change an input image.
# How to define default parameters
1. cal_PDF / cal_CDF function 
This function generates PDF/CDF data of the image which passed as a parameter.
In this code, return type of cal_PDF / cal_CDF function is "float *", so parameter should be a grayscale image.
After using this function, the memory must be released by the **free** function.
2. fopen_s function(<file address to open> ,<text file name> ,<read/write option> )
First parameter is a file address to open. Second parameter is a text file name to save data. Last parameter is a read/write option.
In this code, read/write option is "w+". 
In this option, if text file(passed as a second parameter) already exists, f_open_s function deletes the existing content and create a new one. 
Otherwise, f_open_s function create a new file named <text file name>.
