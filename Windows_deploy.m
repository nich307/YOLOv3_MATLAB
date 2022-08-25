%% Generate oneDNN code for Windows Using MATLAB Coder
% To generate a Windows executable that can be deployed  on to a Intel CPU/GPU 
% target, create a MATLAB code configuration object for generating an executable.
cfg = coder.config('exe');
cfg.TargetLang = 'C++';
cfg.GenerateReport = true;
cfg.DeepLearningConfig = coder.DeepLearningConfig('mkldnn');
cfg.GenerateExampleMain = 'GenerateCodeAndCompile';
%cfg.CustomSource  = fullfile('main.cu');

%% 
codegen('-config ',cfg,'yolov3_onednn', '-report');