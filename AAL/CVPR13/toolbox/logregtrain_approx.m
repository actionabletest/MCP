function w=logregtrain(data,w0,lambda,maxiter)
%
% logistic regression training
%
% Y. Guo
% Jan. 8, 2006
%
% data: t*(d+1), the last column is the label column

[t,d]=size(data);
x=[data(:,1:d-1),ones(t,1)]';
y=data(:,d)';

v=unique(y);
classnum=size(v,2);
if classnum<=2 & max(v)>1
   y=3-2*y;
end

if classnum<=2
%binary classification
   %w0 = zeros(d,1);
   xy = scale_cols(x,y); %note, x is d*n matrix, y is 1*n vector
   [w,run] = logreg_newton(xy,w0,lambda,maxiter);  %0.2

else
%multiclass classification
   y=repmat(y,classnum,1);
   ytmp=[1:classnum]';
   ytmp=repmat(ytmp,1,size(y,2));
   y=(y==ytmp);
   
   %w0=zeros(classnum,d);
   w=mlogreg_newton(x,y,w0,lambda,maxiter);
end

