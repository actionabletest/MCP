clear;
startup;

% the myscript.m file is uesd to get the mutual information measure results
% for the test set based on the precomputed 300 dimensional features. 

% the computation takes most of the time during our experiment, so if you
% don't want to take so much time, you should use our precomputed mutual
% information results for different data sets .

% if you want to get the mutual information measure results for the test
% set by yourself, then you should do these following modifications:
%    1. one directory to load the precomputed 300 dimensional features for
%    the whole test set ---- argument of load
%    2. one directory to save the mutual information measure results
%    ----argument of save


load('cifar\cifar_bim-b_compound8_finalFeatures.mat');  
psi = psi.';
[ut,features_d] = size(psi);
options.KernelType = 'Gaussian';
options.t = 2; %parameters of kernel 
uK = getkernel(psi(:,1:features_d), psi(:,1:features_d),options);
U = uK;
%invU = inv(U+1e-8*eye(ut));
invU = pinv(U);
mutualinfo = {};
interval = 0 ;
for i = 1:ut
    t1 = clock;
    mutualinfo{i} = getmutualinfo(i, U, invU);
    disp(i);
    disp(mutualinfo{i});
    t2 = clock;
    interval = interval + etime(t2,t1);
    disp(etime(t2,t1));
    disp(interval);
end
save('cifar\cifar_bim-b_compound8_mutualinfo','mutualinfo');
 








