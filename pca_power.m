%Author Ashiq Imran
function [ output_args ] = pca_power(varargin )
X = load(varargin{1});
Y = load(varargin{2});
M = str2num(varargin{3});
k = str2num(varargin{4});
m = size(X,1);
n = size(X,2);
A = X(:,1:n-1);
B = Y(:,1:n-1);
u = zeros(n-1,M);
for d = 1:M
    S = cov(A);
    b = rand(n-1,1);  %b0 set as random vector
    for i = 1:k
        val = S*b;
        b = val/norm(val);
    end
    u(:,d) = b;
    fprintf('Eigenvector %d\n',d);
    for l = 1:n-1
        fprintf('%3d: %.4f\n',l,u(l,d));
    end
    A(:,:) = A(:,:) - A(:,:)*u(:,d)*u(:,d)';    
end
fprintf('\n');
R = B*u;
sz = size(R,1);
for l = 1:sz
    fprintf('Test object %d\n',l-1);
    for d = 1:M
        fprintf('%3d: %.4f\n',d,R(l,d));
    end
end
