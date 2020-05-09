% if you want to compute the intermediate 300 dimensional features for the 
% whole test set by yourself. then :
% 1. you should create two folders under testSet, such as:
%       mnist_bim-a_compound8  ---you should provide the .m file which
%   stores the whole test set; such as mnist: 10000*28*28
%       mnist_bim-a_compound8_Features  ---- to store the temporary
%       results, you can delete it after having getted the results.
% 2. you should modify:
%        directories
%               the initial test set .m file 
%               the temprary directory
%               the final 300 dimensional features .m file 
%        arguments of reshape function (28,28 for mnist, 32,32,3 for cifar and svhn )
%        get current_descrs from descrs (761 for mnist, 1105 for cifar and svhn)
clear;
% use load function to load the whole test set 
% so if you want to compute the 300 dimensional feature for each test input for different test sets
% you should provide the initial test set file(.m)
load("testSet\mnist_bim-a_compound8\mnist_bim-a_compound8.mat");



t1 =clock;
descrs = {} ;
for i = 1:10000
    current_testImage = x_test(i,:,:);
    % use reshape function to recover the test image
    % for mnist£¬ reshape(current_testImage,28,28);
    % for cifar and svhn, reshape(current_testImage,32,32,3);
    % to visual and validate the test image, you can use validate.m
    re_current_testImage = reshape(current_testImage,28,28);
    % use getDenseSIFT to compute Dense SIFT descriptors for each test input 
    current_features = getDenseSIFT(re_current_testImage);
    current_features_descr = current_features.descr;
    descrs{i} = current_features_descr;
end
descrs = cat(2,descrs{:});
descrs = single(descrs) ;
% the variable descrs keeps all the Dense SIFT descriptors of the whole test set



% use vl_kmeans to get 300 cates
vocab = vl_kmeans(descrs, 300, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ; 
vocab_kdtree =  vl_kdtreebuild(vocab) ;

for i = 1:10000
% we should split the Dense SIFT descriptors for each test input from all the descriptors
% we should use differnet values to get current_descrs.
% which is 761 for mnist and 1105 for cifar and svhn
current_descrs = descrs(:,(1+(i-1)*761):(i*761));
[vocab_index,distance] = vl_kdtreequery(vocab_kdtree,vocab, ...
                                     single(current_descrs), ...
                                     'MaxComparisons', 50) ;
% the directory should be modified by yourself. it's the temporal results, after getting results, you can delete.                                
% will save vocabIndex for each test input under 'testSet\mnist_bim-a_compound8_Features'
save(['testSet\mnist_bim-a_compound8_Features\image' num2str(i) '_vocabIndex'],'vocab_index');
end


psi = {};
for i = 1:10000   
    % load vocabIndex for each test input
    load(['testSet\mnist_bim-a_compound8_Features\image' num2str(i) '_vocabIndex'],'vocab_index');
    current_z =  vl_binsum(zeros(300,1), 1, double(vocab_index)) ;
    current_z = sqrt(current_z);
    current_z = current_z / max(sqrt(sum(current_z.^2)), 1e-12) ;
    psi{i} = current_z(:) ;    
end
psi = cat(2,psi{:});
% save the final 300 dimensional features for the whole test set 
save('testSet\mnist_bim-a_compound8_finalFeatures', 'psi') ;
t2 = clock;
etime(t2,t1);




        

