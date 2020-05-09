function invBi=calInvSubmat(B, invB, id)
%
% Y. Guo
% Oct. 30, 2009
%
t = size(B,1);
u = [-1; zeros(t-1,1)];
oids = [1:id-1, id+1:t];
v = [B(id,id)-1, B(id,oids)]';
w = [0; B(oids,id)];
invA = invB([id, oids],[id, oids]);
invA1 = invA - (invA*u*v'*invA)/(1+v'*invA*u);
invA2 = invA1 - (invA1*w*u'*invA1)/(1+u'*invA1*w);
invBi = invA2(2:t,2:t);

