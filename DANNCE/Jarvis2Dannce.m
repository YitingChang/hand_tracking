% This script is to convert camera calibration parameters from JARVIS to DANNCE
% 
% JARVIS calibration tool uses OpenCV to perform camera calibration.
% https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
%
% MATLAB provides a function (cameraIntrinsicsFromOpenCV) to convert camera intrinsic parameters 
% from OpenCV to MATLAB.
% https://www.mathworks.com/help/vision/ref/cameraintrinsicsfromopencv.html
% 

yml_path = 'C:\Users\Yiting\Desktop\Calibration_Set_231129\camTo.yaml';
cd 'C:\Users\Yiting\Desktop\Calibration_Set_231129'
filename = 'camTo.yaml';
cameraParams = readCVyaml.helperReadYAML(filename);
