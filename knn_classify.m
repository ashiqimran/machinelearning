%Author Ashiq Imran
function [ output_args ] = knn_classify(varargin )
A = load(varargin{1});
B = load(varargin{2});
K = str2num(varargin{3});
nB = size(B,2);
nA = size(A,2);
cls = A(:,nA);
mA = size(A,1);
mB = size(B,1);
c  = unique(cls);
result = zeros(mB,1);
MEANA = zeros(1,nA-1);
STDA = zeros(1,nA-1);
X = A(:,1:nA-1);
Y = B(:,1:nB-1);
for d = 1:nA-1
    MEANA(1,d) = mean(X(:,d));
    STDA(1,d) = std(X(:,d),1);
    X(:,d) = (X(:,d) - MEANA(1,d))./STDA(1,d);
    Y(:,d) = (Y(:,d) - MEANA(1,d))./STDA(1,d);
end

overall_accuracy = 0;
for i = 1:mB
    dist = sum((Y(i,:)-X).^2,2);
    
    [M, indices] = sort(dist,'ascend');
    
    N = hist(cls(indices(1:K)), c);
    classes = c(find(N == max(N)),1);
    indices = find(classes == B(i,nB));
    accuracy = 0;
    result(i) = classes(randi([1 size(classes,1)],1));
    if(size(indices,1) > 0)
        accuracy = 1/size(classes,1);
        overall_accuracy = overall_accuracy + accuracy;
    end
    fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',i-1, result(i),B(i,nB),accuracy);
end
fprintf('classification accuracy=%6.4f\n', overall_accuracy/mB);