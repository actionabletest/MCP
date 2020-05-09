function minfo=getmutualinfo(yids,K, invK)
%
% compute mutual information between data(yids) and the rest of data
%

n = size(K,1);
yn = length(yids);

SigY = diag(K);
SigY = SigY(yids);
SigYU = SigY;
% matlabpool open;
for i = 1:yn
	id = yids(i);
	uid = [1:id-1, id+1:n];
	invSigUU = calInvSubmat(K,invK,id); 
	SigYU(i) = SigYU(i) - K(id,uid)*invSigUU*K(uid,id);  
end
% matlabpool close;
minfo = 0.5*log2(SigY./SigYU);

