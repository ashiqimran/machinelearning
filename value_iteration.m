%Author Ashiq Imran
function [ output_args ] = value_iteration(varargin)
    file = fopen(varargin{1});
    reward = str2double(varargin{2});
    gamma = str2double(varargin{3});
    K = str2double(varargin{4});
    tline = fgets(file);
    n = 0;
    m = 0;
    while ischar(tline)
        n = n+1;
        m = size(strsplit(tline,','),2);
        tline = fgets(file);
    end
    fclose(file);
    M = zeros(n,m);
    file = fopen(varargin{1});
    tline = fgets(file);
    n=1;
   
    while ischar(tline)
        A = strsplit(tline,',');
        for i=1:m
            a = A{1,i};
            if a(1,1) == '.'
                M(n,i) = 2;
            elseif a(1,1) == 'X'
                    M(n,i) = 3;
            else
                M(n,i) = str2double(a);
            end
        end
        n = n+1;
        tline = fgets(file);
    end
    fclose(file);
    R = reward;
    Unew = zeros(size(M));
    N = size(M,1);
    C = size(M,2);
    for i = 1:K+1
        U = Unew;
        for ii = 1:N
            for jj = 1:C      
                if(M(ii,jj) == 1)
                   Unew(ii,jj) = 1;
                elseif(M(ii,jj) == -1)
                   Unew(ii,jj) = -1;
                elseif(M(ii,jj) == 3)
                   Unew(ii,jj) = 0;
                else
                    Unew(ii,jj) = R + gamma*calculateProb(M,ii,jj,U);
                end
            end
        end
    end
    for i=1:N
        for j=1:C
            if j==1
                fprintf('%6.3f',U(i,j));
            else
                fprintf(',%6.3f',U(i,j));
            end
        end
        fprintf('\n');
    end
end

function mx = calculateProb(M,i,j,U)
mx = -Inf;
sum = 0;
row = size(M,1);
col = size(M,2);
%left
if(j==1 | M(i,j-1) == 3) 
    sum = sum + 0.8*U(i,j);
else
    sum = sum + 0.8*U(i,j-1);
end
if(i==1 | M(i-1,j) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i-1,j);
end
if(i == row | M(i+1,j) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i+1,j);
end
mx = max(mx,sum);

sum = 0;
%right
if(j==col | M(i,j+1) == 3) 
    sum = sum + 0.8*U(i,j);
else
    sum = sum + 0.8*U(i,j+1);
end
if(i==1 | M(i-1,j) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i-1,j);
end
if(i == row | M(i+1,j) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i+1,j);
end

mx = max(mx,sum);

sum = 0;
%up
if(i==1 | M(i-1,j) == 3) 
    sum = sum + 0.8*U(i,j);
else
    sum = sum + 0.8*U(i-1,j);
end
if(j==1 | M(i,j-1) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i,j-1);
end
if(j==col | M(i,j+1) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i,j+1);
end
mx = max(mx,sum);

sum = 0;
%down
if(i==row | M(i+1,j) == 3) 
    sum = sum + 0.8*U(i,j);
else
    sum = sum + 0.8*U(i+1,j);
end
if(j==1 | M(i,j-1) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i,j-1);
end
if(j==col | M(i,j+1) == 3)
    sum = sum + 0.1*U(i,j);
else
    sum = sum + 0.1*U(i,j+1);
end

mx = max(mx,sum);
end

