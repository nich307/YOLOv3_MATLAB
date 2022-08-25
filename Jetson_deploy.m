%% Face Detection on NVIDIA JETSON
% This example shows how to generate and deploy a CUDA(R) executable for a
% face detection application that uses deep learning. It uses the GPU 
% Coder(TM) Support Package for NVIDIA(R) GPUs to deploy the executable on  
% the NVIDIA JETSON(TM) platform. This example performs code generation on the 
% host computer and builds the generated code on the target platform by 
% using remote build capability of the support package.

% Copyright 2019 The MathWorks, Inc. 

%% Prerequisites
% *Target Board Requirements*
%
% * NVIDIA JETSON Nano embedded platform. 
% * Ethernet crossover cable to connect the target board and host PC (if 
% the target board cannot be connected to a local network).
% * NVIDIA CUDA toolkit installed on the board.
% * NVIDIA cuDNN library (v5 and above) on the target.
% * OpenCV 3.0 or higher library on the target for reading and displaying
% images/video
% * Environment variables on the target for the compilers and libraries. 
% For information on the supported versions of the compilers and libraries 
% and their setup, see <matlab:web(fullfile(docroot,'supportpkg/nvidia/ug/install-and-setup-prerequisites.html'))
%  installing and setting up prerequisites for NVIDIA boards>.
%
% *Development Host Requirements*

%% Connect to the NVIDIA Hardware
% The GPU Coder Support Package for NVIDIA GPUs uses an SSH connection over 
% TCP/IP to execute commands while building and running the generated CUDA 
% code on the Jetson platform. You must therefore connect the
% target platform to the same network as the host computer or use an 
% Ethernet crossover cable to connect the board directly to the host
% computer. Refer to the NVIDIA documentation on how to set up and
% configure your board.
%
% To communicate with the NVIDIA hardware, you must create a live hardware
% connection object by using the <matlab:doc('jetson') jetson> function. You
% must know the host name or IP address, username, and password of the 
% target board to create a live hardware connection object. For example,
%
   hwobj = jetson('sha-xavier','xavier','matlab');
   %hwobj = jetson('sha-nano','nano','matlab');

%% Verify the GPU Environment
% Use the <matlab:doc('coder.checkGpuInstall') coder.checkGpuInstall> function
% and verify that the compilers and libraries needed for running this example
% are set up correctly.
% 
envCfg = coder.gpuEnvConfig('jetson');
envCfg.DeepLibTarget = 'tensorrt';
envCfg.DeepCodegen = 1;
envCfg.HardwareObject = hwobj;
coder.checkGpuInstall(envCfg);


%% Generate CUDA Code for the Target Using GPU Coder
% To generate a CUDA executable that can be deployed  on to a NVIDIA 
% target, create a GPU code configuration object for generating an executable.
cfg = coder.gpuConfig('exe');
cfg.GenerateReport = true;
cfg.Hardware = coder.hardware('NVIDIA Jetson');
%cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
cfg.DeepLearningConfig = coder.DeepLearningConfig('tensorrt');
cfg.DeepLearningConfig.DataType = 'fp16';
cfg.GpuConfig.ComputeCapability = '7.0';
cfg.Hardware.BuildDir = '~/remoteBuildDir';
cfg.GpuConfig.SelectCudaDevice = 0;
cfg.GpuConfig.MallocMode = "unified";
cfg.GenerateExampleMain = 'GenerateCodeAndCompile';
%cfg.CustomSource  = fullfile('main.cu');

%% 
% To generate CUDA code, use the <matlab:doc('codegen') codegen> function 
% and pass the GPU code configuration along with
% |sobelEdgeDetection| entry-point function. After the  code generation takes place on 
% the host, the generated files are copied over and built on the target.
%
   codegen('-config ',cfg,'yolov3_detection', '-report');

%% Run the Sobel Edge Detection on the Target
% Run the generated executable on the target.
%
   pid = hwobj.runApplication('yolov3_detection');
%

%%
% A window opens on the target hardware display showing the Sobel edge 
% detection output of the live webcam feed.
%
% <<sobel_deploy_out_rsz.png>>

%% Cleanup
% Remove the files and return to the original folder.
% 
%   cleanup

%% Summary
%
% This example introduced an application where Sobel edge detection 
% application is running on the NVIDIA hardware on the live webcam feed and
% displaying the output on the native display.

displayEndOfDemoMessage(mfilename)
