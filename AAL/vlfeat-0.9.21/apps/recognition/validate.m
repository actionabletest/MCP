clear;
load("testSet\svhn_bim-a_compound8\svhn_bim-a_compound8");
testImage1 = x_test(2349,:,:);
Reshape_testImage1 = reshape(testImage1,32,32,3);
current_features = getDenseSIFT(Reshape_testImage1);
% size(testImage1);
%y = y_test(2349);
%imshow(Reshape_testImage1);