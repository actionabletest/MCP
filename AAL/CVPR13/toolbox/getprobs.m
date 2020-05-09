function probs=getprobs(data,w,classnum)
%
% Y. Guo
% Jan. 8, 2006
%
% data: t*(d+1), the last column is the label column

[t,d]=size(data);

x=[data(:,1:d-1),ones(t,1)]';

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
