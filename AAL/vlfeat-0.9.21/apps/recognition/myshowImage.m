% clear;
% load("cifar-10-batches-mat\test_batch.mat");
% descrs = {} ;
% for i = 1:10000
% image_vector = data(i,:);
% image_red_vector = image_vector(1:1024);
% image_red_matrix = reshape(image_red_vector,32,32);
% image_red_matrix = image_red_matrix' ;
% image_green_vector = image_vector(1025:2048);
% image_green_matrix = reshape(image_green_vector,32,32);
% image_green_matrix = image_green_matrix' ;
% image_blue_vector = image_vector(2049:3072);
% image_blue_matrix = reshape(image_blue_vector,32,32);
% image_blue_matrix = image_blue_matrix' ;
% final_image = cat(3,image_red_matrix,image_green_matrix,image_blue_matrix);
% % imshow(final_image);
% current_features = getDenseSIFT(final_image);
% current_features_descr = current_features.descr;
% descrs{i} = current_features_descr;
% % imshow(final_image);
% end 
% % descrs = cat(2,descrs{:});
% % descrs = single(descrs) ;

% vocab_kdtree =  vl_kdtreebuild(vocab) ;
% for i = 1:10000
% current_descrs = descrs(:,(1+(i-1)*1105):(i*1105));
% [vocab_index,distance] = vl_kdtreequery(vocab_kdtree,vocab, ...
%                                      single(current_descrs), ...
%                                      'MaxComparisons', 50) ;
%                                   
% save(['cifar300/image' num2str(i) '_vocabIndex'],'vocab_index');
% end


% 
% psi = {};
% for i = 1:10000    
%     load(['cifar300/image' num2str(i) '_vocabIndex'],'vocab_index');
%     current_z =  vl_binsum(zeros(300,1), 1, double(vocab_index)) ;
%     current_z = sqrt(current_z);
%     current_z = current_z / max(sqrt(sum(current_z.^2)), 1e-12) ;
%     psi{i} = current_z(:) ;    
% end
% psi = cat(2,psi{:});
% save('cifar_results/hists300', 'psi') ;



