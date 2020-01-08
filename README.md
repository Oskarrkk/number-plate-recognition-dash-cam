## Number Plate Detection and Recognition
It is desktop application for number plate detection and recognition in dash cam videos.
	
## Technologies
Project is created with:
* PyQt5 - to create gui
* OpenCV and NumPy
* skimage - to get algorithms which calculate hog and sift features
* scikit-learn - for machine learning purpose

## Presentation - How It Works

The concept of plotting GWM on-line was relatively simple. It assumed that a buffer of a certain size would be created, which would contain samples collected from the microphone. When new 
samples arrive, they are placed in the buffer in place of the old ones. Samples in the buffer are input samples for the FFT algorithm. 
After appropriate transformations, the output from FFT is plotted as PSD on the chart. This process is repeated all the time with the arrival of new samples.

![](video.gif)
