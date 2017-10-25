%Author Ashiq Imran
function [ output_args ] = linear_regression(varargin )
A = load(varargin{1});
degree = str2num(varargin{2});
M = size(A,1);
N = size(A,2);
X = A(:,1:N-1);
XT = X.';
T = A(:,N);
for i = 1:M
    if(T(i)==1)
        t(i) = 1;
    else
        t(i) = 0;
    end
end
t = t';
Dim = size(X,2);
sz = size(X,1);
Phi = ones(sz,degree*Dim + 1);
W = zeros(degree*Dim+1,1);
WN = zeros(degree*Dim+1,1);
szPhi = size(Phi,2);
R = zeros(sz,sz);
if degree == 1
   Phi(:,2:szPhi) = X(:,:);
elseif degree == 2
    for i = 1:szPhi/2
            Phi(:,2*i) = X(:,i);
            Phi(:,2*i+1) = power(X(:,i),2);
    end
end
PhiT = Phi.';
while(1)
a = W.'*PhiT;
y = (1./(1+exp(-a)))';
E = PhiT*(y-t);
for i = 1: M
    R(i, i) = y(i)*(1-y(i));
end
H = PhiT*R*Phi;
WN = W - pinv(H)*E;
aN = WN.'*PhiT;
yN = (1./(1+exp(-aN)))';
EN = PhiT*(yN-t);
S = sum(abs(WN-W));
W = WN;
delE = abs(EN-E);
if (S < 0.001 | delE < 0.001)
    break;
end
end
%**********Printing W*********
for i = 0:size(W,1)-1
    fprintf('w%1d=%.4f\n',i, W(i+1));
end
%*********************************TESTING**********************************
B = load(varargin{3});
MB = size(B,1);
NB = size(B,2);
XB = B(:,1:NB-1);
TB = B(:,NB);
DimB = size(XB,2);
szB = size(XB,1);
PhiB = ones(szB,degree*DimB + 1);
for i = 1:MB
    if(TB(i)==1)
        tB(i) = 1;
    else
        tB(i) = 0;
    end
end
% 
szPhiB = size(PhiB,2);
if degree == 1
   PhiB(:,2:szPhiB) = XB(:,:);
elseif degree == 2
    for i = 1:szPhiB/2
            PhiB(:,2*i) = XB(:,i);
            PhiB(:,2*i+1) = power(XB(:,i),2);
    end
end
PhiBT = PhiB.';
aB = W.'*PhiBT;
yB = (1./(1+exp(-aB)))';
overall_accuracy = 0;
for i = 1:szB
    accuracy = 0;
    if(yB(i) >= 0.5)
        predicted_class = 1;
        prob = yB(i);
    else
        predicted_class = 0;
        prob = 1 - yB(i);
        accuracy = 0;
    end
    
    if(predicted_class == tB(i))
        accuracy = 1;
        overall_accuracy = overall_accuracy + accuracy;
    else
        accuracy = 0;
    end
    fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n', i-1,predicted_class,prob, tB(i), accuracy);
end
fprintf('classification accuracy=%6.4f\n',overall_accuracy/MB);




