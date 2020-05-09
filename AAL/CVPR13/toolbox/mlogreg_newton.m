function  w = mlogreg_newton(x,y,w0,lambda,maxiter)
%
% Here, adjusted this function to deal with multiway classification
% Dec 10, 2004
%
% x: d*n
% y: k*n, it is indicator function: y_{i,j}=[label of x^j=i], it is 0 or 1  
% w: k*d, where k is the number of classes, d is the feature num.

if nargin < 4
  lambda = 0;
end
w=w0;

[d,n] = size(x);
[k,d]=size(w);
%convert w into a (k*d,1) vector first
w=reshape(w',k*d,1);

for iter = 1:maxiter
  old_w = w;

  %compute g
  mw=reshape(w,d,k)';
  expx=exp(mw*x); %k*n
  probx=expx./repmat(sum(expx),k,1); %k*n
  g=(y-probx)*x'; %k*d
  g=reshape(g',k*d,1)-lambda*w;

  %compute h
  h=[];
  for i=1:k %index g
      trow=[];
      for j=1:k  %index w
          dgw=probx(i,:).*probx(j,:); %1*n
	  deduct=0;
          if i==j
	     dgw=dgw-probx(i,:);
	     deduct=lambda*eye(d);
	  end   
	  tmp=repmat(dgw,d,1);
	  dgw=(tmp.*x)*x'-deduct;

	  trow=[trow dgw];
      end
      h=[h; trow];
  end

  w=w-h\g;

%   if max(abs(w - old_w)) < 1e-8
 if (max(abs(w - old_w)) < 1e-5)|((max(abs(w - old_w)) < 1e-3 )& iter >100)
    break
  end
end

%reshape w back into k*d matrix

w=reshape(w,d,k)';

if (iter == maxiter) & (maxiter>100)
  warning('not enough iters')
end
