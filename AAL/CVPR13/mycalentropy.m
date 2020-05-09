% mycalentropy.m file to compute the final AdaptAL measure results for the
% testset

% to run the file successfully, you should only modifies three diectories.
% one is the softmax file to load softmax values to compute entropy
% one is the mutualinfo file to load mutual infomation measure results 
% one is the finalResults file to save the final AdaptAL measure results  
all_entropyvec = {};
softmax = load("cifar_softmax\bim-a_compound_cifar_softmax.csv");
for i=2:10001
probs = softmax(i,2:11);
tmp=(probs<1e-15);
entropyvec=- probs.*log2(probs+tmp*1e-15);
entropyvec=sum(entropyvec)'; %t*1
all_entropyvec{i-1} = entropyvec;
end
load("cifar_finalmutualinfo\cifar_bim-a_compound8_mutualinfo.mat")
beta = 0.5 ;
measures = (cell2mat(all_entropyvec).^beta).*((mutualinfo).^(1-beta));
save("temp1","measures");
