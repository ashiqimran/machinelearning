%Author Ashiq Imran
function [] = k_means_cluster(varargin)
A = load(varargin{1});
m = size(A,1);
n = size(A,2);
X = A(:,1:n-1);
K = str2num(varargin{2});
maxIter = str2num(varargin{3});
% number of vectors in X
[vectors_num, dim] = size(X);
C = zeros(K, dim);
clstr = randi([1 K],1,m)';
index = randi([1 size(X,1)],1,K)';
C = X(index,:);
error = 0;
 for n=1:vectors_num
        % find closest center to current input point
        for k=1:K
            C(k, :) = mean(X(find(clstr == k), :));
        end

        error = error + norm(X(n,:) - C(clstr(n),:), 2);
 end
 fprintf('After initialization: error = %.4f\n',error);

for iter = 1:maxIter

    for n=1:vectors_num
        % find closest center to current input point
        minIdx = 1;
        minVal = norm(X(n,:) - C(minIdx,:), 1);
        for j=1:K
            dist = norm(C(j,:) - X(n,:), 1);
            error = error + dist;
            if dist < minVal
                minIdx = j;
                minVal = dist;

            end
        end

        I(n) = minIdx;
    end
    % compute centers
    for k=1:K
        C(k, :) = mean(X(find(I == k), :));
    end
    % compute  error
    error = 0;
    for idx=1:vectors_num
        error = error + norm(X(idx, :) - C(I(idx),:), 2);
    end

    fprintf('After iteration %d: error = %.4f\n',iter,error);
end
