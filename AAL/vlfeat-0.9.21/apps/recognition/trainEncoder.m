function encoder = trainEncoder(images, varargin)
% TRAINENCODER   Train image encoder: BoVW, VLAD, FV
%   ENCODER = TRAINENCOER(IMAGES) trains a BoVW encoder from the
%   specified list of images IMAGES.
%
%   TRAINENCODER(..., 'OPT', VAL, ...) accepts the following options:
%
%   Type:: 'bovw'
%     Bag of visual words ('bovw'), VLAD ('vlad') or Fisher Vector
%     ('fv').
%
%   numPcaDimension:: +inf
%     Use PCA to reduce the descriptor dimensionality to this
%     dimension. Use +inf to deactivate PCA.
%
%   Whitening:: false
%     Set to true to divide the principal components by the
%     corresponding standard deviation s_i.
%
%   WhiteningRegul:: 0
%     When using whitening, divide by s_max * WhiteningRegul + s_i
%     instead of s_i alone.
%
%   Renormalize:: false
%     If true, descriptors are L2 normalized after PCA or
%     whitening.
%
%
%   Subdivisions:: []
%     A list of spatial subdivisions. Each column is a rectangle
%     [XMIN YMIN XMAX YMAX]. The spatial subdivisions are
%
%   Layouts:: {'1x1'}
%     A list of strings representing regular spatial subdivisions
%     in the format MxN, where M is the number of vertical
%     subdivisions and N the number of horizontal ones. For
%     example {'1x1', 2x2'} uses 5 partitions: the whole image and
%     four quadrants. The subdivisions are appended to the ones
%     specified by the SUBDIVISIONS option.
%
%   ReadImageFn:: @readImage
%     The function used to load an image.
%
%   ExtractorFn:: @getDenseSIFT
%     The function used to extract the feature frames and
%     descriptors from an image.

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.type = 'bovw' ;
opts.numWords = [] ;
opts.seed = 1 ;
opts.numPcaDimensions = +inf ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.numSamplesPerWord = [] ;
opts.renormalize = false ;
opts.layouts = {'1x1'} ;
opts.geometricExtension = 'none' ;
opts.subdivisions = zeros(4,0) ;
opts.readImageFn = @readImage ;
opts.extractorFn = @getDenseSIFT ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

for i = 1:numel(opts.layouts)
  t = sscanf(opts.layouts{i},'%dx%d') ;
  m = t(1) ;
  n = t(2) ;
  [x,y] = meshgrid(...
    linspace(0,1,n+1), ...
    linspace(0,1,m+1)) ;
  x1 = x(1:end-1,1:end-1) ;
  y1 = y(1:end-1,1:end-1) ;
  x2 = x(2:end,2:end) ;
  y2 = y(2:end,2:end) ;
  opts.subdivisions = cat(2, opts.subdivisions, ...
    [x1(:)' ;
     y1(:)' ;
     x2(:)' ;
     y2(:)'] ) ;
end

if isempty(opts.numWords)
    switch opts.type
      case {'bovw'}
          % 我需要改一下
        opts.numWords = 1024 ;
      case {'fv'}
        opts.numWords = 64 ;
        opts.numPcaDimensions = 80 ;
      case {'vlad'}
        opts.numWords = 64 ;
        opts.numPcaDimensions = 100 ;
        opts.whitening = true ;
        opts.whiteninRegul = 0.01 ;
      otherwise
        assert(false) ;
    end
end

if isempty(opts.numSamplesPerWord)
    switch opts.type
      case {'bovw'}
          %  numSamplesPerWord什么意思？？？
        opts.numSamplesPerWord = 200 ;
      case {'vlad','fv'}
        opts.numSamplesPerWord = 1000 ;
      otherwise
        assert(false) ;
    end
    if opts.lite
      opts.numSamplesPerWord = 10 ;
    end
end

% disp显示变量的值
disp(opts) ;

encoder.type = opts.type ;
encoder.subdivisions = opts.subdivisions ;
encoder.readImageFn = opts.readImageFn ;
encoder.extractorFn = opts.extractorFn ;
encoder.numWords = opts.numWords ;
encoder.renormalize = opts.renormalize ;
encoder.geometricExtension = opts.geometricExtension ;

%% Step 0: obtain sample image descriptors
% 获取图片的数量
numImages = numel(images) ;
% 需要每一张图片的关键点特征向量的数量  ceil(X) 将 X 的每个元素四舍五入到大于或等于该元素的最接近整数。
numDescrsPerImage = ceil(opts.numWords * opts.numSamplesPerWord / numImages) ;
parfor i = 1:numImages
  fprintf('%s: reading: %s\n', mfilename, images{i}) ;
  % 读取图片 
  im = encoder.readImageFn(images{i}) ;
  w = size(im,2) ;
  h = size(im,1) ;
  % 获取图片的dense sift features
  features = encoder.extractorFn(im) ;
  % 设定初始状态的好处是，只需要保存那时的初始状态再运行一遍程序你就可以重现之前的计算过程和结果
  randn('state',0) ;
  rand('state',0) ;
  % size(features.descr,2)表明当前图片有多少个关键点（特征向量descr），
  % sel是从一张图片中的所有特征向量中随机抽取一部分特征向量所得到的特征向量集合   
  sel = vl_colsubset(1:size(features.descr,2), single(numDescrsPerImage)) ;
  descrs{i} = features.descr(:,sel) ;
  frames{i} = features.frame(:,sel) ;
  frames{i} = bsxfun(@times, bsxfun(@minus, frames{i}(1:2,:), [w;h]/2), 1./[w;h]) ;
end
% 抽取到用于训练编码器的数据
descrs = cat(2, descrs{:}) ;
frames = cat(2, frames{:}) ;

% %% Step 1 (optional): learn PCA projection
% if opts.numPcaDimensions < inf || opts.whitening
%   fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
%   encoder.projectionCenter = mean(descrs,2) ;
%   x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
%   X = x*x' / size(x,2) ;
%   [V,D] = eig(X) ;
%   d = diag(D) ;
%   [d,perm] = sort(d,'descend') ;
%   d = d + opts.whiteningRegul * max(d) ;
%   m = min(opts.numPcaDimensions, size(descrs,1)) ;
%   V = V(:,perm) ;
%   if opts.whitening
%     encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
%   else
%     encoder.projection = V(:,1:m)' ;
%   end
%   clear X V D d ;
% else
%   encoder.projection = 1 ;
%   encoder.projectionCenter = 0 ;
% end
% descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;
% if encoder.renormalize
%   descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
% end
% 
% 
% %% Step 2 (optional): geometrically augment the features
% 
% descrs = extendDescriptorsWithGeometry(opts.geometricExtension, frames, descrs) ;

%% Step 3: learn a VQ or GMM vocabulary
% dimension 表示特征向量维度，这里我们不做改变 。
dimension = size(descrs,1) ;
% numDescriptors 表示用于训练编码器的特征向量训练集的数目
numDescriptors = size(descrs,2) ;

switch encoder.type
  case {'bovw', 'vlad'}
    vl_twister('state', opts.seed) ;
    encoder.words = vl_kmeans(descrs, opts.numWords, 'verbose', 'algorithm', 'elkan') ;
    encoder.kdtree = vl_kdtreebuild(encoder.words, 'numTrees', 2) ;

  case {'fv'} ;
    vl_twister('state', opts.seed) ;
    if 1
      v = var(descrs')' ;
      [encoder.means, encoder.covariances, encoder.priors] = ...
          vl_gmm(descrs, opts.numWords, 'verbose', ...
                 'Initialization', 'kmeans', ...
                 'CovarianceBound', double(max(v)*0.0001), ...
                 'NumRepetitions', 1) ;
    else
      addpath lib/yael/matlab
      [a,b,c] = ...
          yael_gmm(descrs, opts.numWords, 'verbose', 2) ;
      encoder.priors = single(a) ;
      encoder.means = single(b) ;
      encoder.covariances = single(c) ;
    end
end
