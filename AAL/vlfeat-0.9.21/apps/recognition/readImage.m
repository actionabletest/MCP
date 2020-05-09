function [im, scale,intial_size] = readImage(imagePath)
% READIMAGE   Read and standardize image
%    [IM, SCALE] = READIMAGE(IMAGEPATH) reads the specified image file,
%    converts the result to SINGLE class, and rescales the image
%    to have a maximum height of 480 pixels, returing the corresponding
%    scaling factor SCALE.
%
%    READIMAGE(IM) where IM is already an image applies only the
%    standardization to it.

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ischar(imagePath)
  try
    im = imread(imagePath) ;
  catch
    error('Corrupted image %s', imagePath) ;
  end
else
  im = imagePath ;
end

intial_size = size(im);
% Convert image to single precision
im = im2single(im) ;


scale = 1 ;
if (size(im,1) > 480)
  scale = 480 / size(im,1) ;
  % 将 A 的长宽大小缩放 scale 倍之后的图像;如果 A 有两个以上维度，则 imresize 只调整前两个维度的大小。
  im = imresize(im, scale) ;
  %这是什么意思？？ --- 确保图像中的所有像素值在0.0000~1.0000之间
  im = min(max(im,0),1) ;
end

