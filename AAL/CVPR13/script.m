clear all;
dbstop if error
startup;

load('voc07_densesift.mat');
load('voc_labelset.mat');
load('voc_unlabelset.mat');
load('voc_testset.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%parameter setting: you can adjust them%%%%%%%%%%%%%%%
options.KernelType = 'Gaussian';
options.t = 2; %parameters of kernel
% subsampling.method
%   0: use all
%	1: top most uncertain ones
%	2: clustering
%	3: random selection
% subsampling.num
subsampling.method = 3;
subsampling.num = 200; % sample set size of unlabeled data
addnum = 100; % selection iterations

%%==========================method=========================
fprintf('Adaptive Active Learning begins...\n');
allbeta = 0:0.1:1;
[adaptiveresult, ~]=ActiveCombSample(labelset,unlabelset,testset,allbeta,...
    options, addnum, subsampling, 1);