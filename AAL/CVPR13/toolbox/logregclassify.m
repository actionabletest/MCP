function [accu,yvec]=logregclassify(data,w,classnum)
%
% logistic regression classification
%
% Y. Guo
% Jan. 8, 2006
%
% data: t*(d+1), the last column is the label column

[t,d]=size(data);

x=[data(:,1:d-1),ones(t,1)]';
y=data(:,d)';

v=unique(y);
%classnum=size(v,2);
if classnum<=2 & max(v)>1
   y=3-2*y;
end

if classnum<=2
%binary classification
   s=w'*x; %1*t
   p=1./(1+exp(-s)); %if y=1
   yvec=(p>=0.5);
   yvec=2*yvec-1;
   accu=sum(y==yvec)/t;

else
%multiclass classification
   expx=exp(w*x); %classnum*t
   probx=expx./repmat(sum(expx),classnum,1);
   [p,yvec]=max(probx,[],1); %1*t
   accu=sum(yvec==y)/t;
end

