%Author Ashiq Imran
function [ output_args ] = neural_network( varargin )
A = load(varargin{1});
B = load(varargin{2});
L = str2double(varargin{3});
U = str2double(varargin{4});
R = str2double(varargin{5});
n = size(A,1);
m = size(A,2);
nB = size(B,1);
mB = size(B,2);
D = m - 1;
DB = mB - 1;
A(:,D+1) = A(:,D+1) + 1;
B(:,DB+1) = B(:,DB+1) + 1;
c = unique(A(:,m));
szc = size(c,1);
t = zeros(n,szc);
for i = 1:n
    t(i,A(i,D+1)) = 1;
end
%finding max Value in training set
max_val = max(max(A(:,1:D)));
X = A(:,1:D);
X = X./max_val;
Y = B(:,1:DB);
Y = Y./max_val;
A = [X A(:,D+1)];
B = [Y B(:,D+1)];
szM = max(max(U,D+1),szc)+1;
W = zeros(szM,szM,L-1);
Wnew = zeros(szM,szM,L-1);
%intialization
eta = 1;

for i = 1: L-1
    W(:,:,i) = -0.05 + (0.05+0.05).*rand(szM,szM);
end
%compute outputs
for r = 1:R
    for x = 1:n
        z = zeros(L,szM);
        a = zeros(L,szM);
        z(1,1:D) = A(x,1:D);
        for l = 2:L
           percep_size = U;
           if l == L
               percep_size = szc;
           end
           p_size = U;
           if l==2
               p_size = D;
           end
           
           
           for j = 1:percep_size
               sum = 0;
               for i = 1:p_size 
                    sum = sum + W(j,i+1,l-1)*z(l-1,i); 
               end
               sum = sum + W(j,1,l-1); %for bias weight
               a(l,j) = sum;
               z(l,j) = (1./(1+exp(-sum)));
           end                 
        end
        
        %Update weights in output layer
        delta = zeros(L,szM);
        p_unit = D;
        if L > 2
            p_unit = U;
        end
        Wnew = W;
         for j = 1:szc
                 delta(L,j) = (z(L,j) - t(x,c(j,1))) * z(L,j) * (1 - z(L,j));
                 for i = 1:p_unit
                     Wnew(j,i+1,L-1) = W(j,i+1,L-1) - (eta*(delta(L,j)*z(L-1,i)));
                 end
                 Wnew(j,1,L-1) = W(j,1,L-1) - (eta*(delta(L,j)*1)); %for bias weight
         end
         %Update weights in hidden layer
         for l = L-1:-1:2  
             for j = 1:U
                 front_size = U;
                 if l == L-1
                     front_size = szc;
                 end
                 sum = 0;
                 for u = 1:front_size
                     sum = sum + delta(l+1,u)*W(u,j+1,l);
                 end
                 delta(l,j) = sum*z(l,j)*(1 - z(l,j));
                 back_size = U;
                 if l==2
                     back_size = D;
                 end
                 for i = 1:back_size
                     Wnew(j,i+1,l-1) = W(j,i+1,l-1) - eta*(delta(l,j)*z(l-1,i));
                 end
                     Wnew(j,1,l-1) = W(j,1,l-1) - eta*delta(l,j)*1;
             end
             
         end
        W = Wnew;

    end
    eta = eta * 0.98;
end
%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%
overall_accuracy = 0;
for y = 1:nB
    
    z = zeros(L,szM);
    a = zeros(L,szM);
    z(1,1:D) = B(y,1:D);
    results = zeros(szc,1);
    for l = 2:L
        percep_size = U;
        if L == 2 || l == L
            percep_size = szc;
        end
        p_size = U;
        if l==2
            p_size = D;
        end
        for j=1:percep_size
            sum = 0;
            for i=1:p_size
                sum = sum + z(l-1,i)*W(j,i+1,l-1);
            end
            sum = sum + W(j,1,l-1); %for bias weight
            a(l,j) = sum;
            z(l,j) = (1./(1+exp(-sum)));
            if l==L
                results(j,1) = z(l,j);
            end
        end
    end
    class_fnd = c(find(results == max(results),1));
    indices = find(class_fnd == B(y,D+1));
    accuracy = 0;
    if size(indices,1) > 0
        accuracy = (1/size(class_fnd,1));
        overall_accuracy = overall_accuracy + accuracy;
    end
    sz = size(class_fnd,1);
    predicted_class = class_fnd( randi([1, sz]),1);
    fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', y-1,predicted_class-1, B(y,mB)-1, accuracy);
 
    
end
fprintf('classification accuracy=%6.4f\n',overall_accuracy/nB);

