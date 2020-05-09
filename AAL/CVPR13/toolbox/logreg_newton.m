function [w,run] = logreg_newton(x,w,lambda,maxiter)
% TRAIN_NEWTON    Train a logistic regression model by Newton's method.
%
% W = TRAIN_NEWTON(X,W) returns maximum-likelihood weights given data and a
% starting guess.
% Data is columns of X, each column already scaled by the output.
% W is the starting guess for the parameters (a column).
% If it fails, try initializing with smaller magnitude W.
% W = TRAIN_NEWTON(X,W,LAMBDA) returns MAP weights with smoothing parameter
% LAMBDA.
%
% Written by Thomas P Minka
%
%x: d*n
%w: d*1

if nargin < 3
  lambda = 0;
end

[d,n] = size(x);

%run.oldlogprob=logProb(x,w) -0.5*lambda*w'*w;
%run.oldw=w;

for iter = 1:maxiter
  old_w = w;
  % s1 is 1 by n
  % s1 = 1-sigma

  s1 = 1./(1+exp(w'*x));
  a = s1.*(1-s1); %A

  g = x*s1' - lambda*w;
  h = scale_cols(x,a)*x' + lambda*eye(d);
  w = w + h\g;
  
%  if nargout > 1
%    run.w(:,iter) = w;
%    run.e(iter) = logProb(x,w) -0.5*lambda*w'*w;
%  end
  
%  run.logprob=logProb(x,w) -0.5*lambda*w'*w;
%  if run.logprob < run.oldlogprob
%     w=run.oldw;    
%     break;
%  else
%     run.oldlogprob=run.oldlogprob
%     run.oldw=w;
%  end
  
%  if max(abs(w - old_w)) < 1e-8
  if (max(abs(w - old_w)) < 1e-5)|((max(abs(w - old_w)) < 1e-3 )& iter >100)
    break
  end
end

run=[];  

if (iter == maxiter) & (maxiter>100)
  iter
  warning('not enough iters')
end
