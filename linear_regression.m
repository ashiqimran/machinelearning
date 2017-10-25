%Author Ashiq Imran
function [ output_args ] = linear_regression(varargin )
A = load(varargin{1});
degree = str2num(varargin{2});
lambda = str2num(varargin{3});
X = A(:,1);
T = A(:,2);
sz = size(X,1);
Phi = ones(sz,degree+1);
I = eye(degree+1);
W = zeros(degree+1,1);
for i = 1:degree
    Phi(:,i+1) = power(X,i);
end
PhiT = Phi.';
Y = pinv(I*lambda + PhiT*Phi)*PhiT;
W = Y*T;
if degree==1
    fprintf('w0=%.4f\nw1=%.4f\nw2=%.4f\n',W(1),W(2),0);
end
if degree==2
    fprintf('w0=%.4f\nw1=%.4f\nw2=%.4f\n', W(1),W(2),W(3));
end

