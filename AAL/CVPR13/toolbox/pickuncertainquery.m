function id=pickuncertainquery(ulist,unlabelset,w,classnum)
%
% Selecting the query that has largest entropy
%
% Y. Guo
% Jan. 8, 2006
%
% data: t*(d+1), the last column is the label column


entropyvec=calentropy(ulist,unlabelset,w,classnum);
[v,id]=max(entropyvec);
