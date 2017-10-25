%Author Ashiq Imran
function [ output_args ] = svd_power(varargin )
X = load(varargin{1});
m = size(X,1);
n = size(X,2);
M = str2num(varargin{2});;
k = str2num(varargin{3});;
U = zeros(m,M);
A = X;
B = X;
V = zeros(n,M);
S = zeros(M);
Sa = zeros(M);
Sb = zeros(M);
for d=1:M
    Sa = A*A';
    b = ones(m,1);
    for i = 1:k
        val = Sa*b;
        b = val/norm(val);
    end
    U(:,d) = b;
    A = (A - ((A'*b)*b')');
end

fprintf('Matrix U:\n');
for l = 1:m
    fprintf('  Row   %d:',l);
    for d = 1:M
        fprintf('%8.4f',U(l,d));
    end
    fprintf('\n');
end
A = X*X';
for d = 1:M
    l = (A*U(:,d))'*U(:,d);
    o = U(:,d)'*U(:,d);
    S(d,d) = sqrt(l/o); %eigen value
end
fprintf('Matrix S:\n');
for l = 1:M
    fprintf('  Row   %d:',l);
    for d = 1:M
        fprintf('%8.4f',S(l,d));
    end
    fprintf('\n');
end
B = X;
for d = 1:M
    Sb = B'*B;
    bv = ones(n,1);
    for i = 1:k
        val = Sb*bv;
        bv = val/norm(val);
    end
    V(:,d) = bv;
    B = (B - B*bv*bv');
end
fprintf('Matrix V:\n');
for l = 1:n
    fprintf('  Row   %d:',l);
    for d = 1:M
        fprintf('%8.4f',V(l,d));
    end
    fprintf('\n');
end
R = U*S*V';
fprintf('Reconstruction (U*S*V''):\n');
for l = 1:m
    fprintf('  Row   %d:',l);
    for d = 1:n
        fprintf('%8.4f',R(l,d));
    end
    fprintf('\n');
end
