function entropyvec=getentropy(data,w,classnum)
%
% compute entropies for the instances 
%
% Y. Guo
% Jan. 8, 2006
%
% data: t*(d+1), the last column is the label column

[t,d]=size(data);

x=[data(:,1:d-1),ones(t,1)]';

%y=data(:,d)';
%v=unique(y);
%classnum=size(v,2);

if classnum<=2
   %binary classification
   s=w'*x; %1*t
   probs(1,:)=1./(1+exp(-s)); %if y=1
   probs(2,:)=1-probs(1,:);  %2*t
else
   %multiclass classification
   expx=exp(w*x); %classnum*t
   probs=expx./repmat(sum(expx),classnum,1);
end

%to avoid log zero
tmp=(probs<1e-15);
entropyvec=- probs.*log2(probs+tmp*1e-15);
entropyvec=sum(entropyvec)'; %t*1

