%Author Ashiq Imran
function [ output_args ] = naive_bayes(varargin )
trainFile = load(varargin{1});
testFile = load(varargin{2});

rowSize = size(trainFile,1);
colSize = size(trainFile,2);

uniqueClasses = unique(trainFile(:,colSize));
NoOfClasses = size(uniqueClasses,1);

type = varargin{3};
testSize = size(testFile,1);
overall_accuracy = 0;

countClass=zeros(size(uniqueClasses,1),1);
for c=1:NoOfClasses
    countClass(c) = size(find(trainFile(:,colSize)==uniqueClasses(c)),1);
end

if strcmp(type, 'histograms')
    %Training
    N = str2num(varargin{4}); %No. of bins
    freqCount = zeros(NoOfClasses,colSize-1,N);
    bound = zeros(NoOfClasses,colSize-1, 3);
    for c = 1:NoOfClasses
        for d = 0:colSize-2
            index = find(trainFile(:,colSize)==uniqueClasses(c));
            V = trainFile(index,d+1);
            sampleSize = size(V,1);
            bound(c,d+1,1) = min(V);
            bound(c,d+1,2) = max(V);
            S = bound(c,d+1,1);
            L = bound(c,d+1,2);
            G = (L-S)/(N-3);
            G = max(G,0.0001);
            for b = 0:N-1
                if b == 0 || b == N-1
                    freqCount(c,d+1,b+1) = 0;
                    P = 0.00;
                    
                else
                    bound(c,d+1,3) = sampleSize;
                    binCount = size( find(V >= S+(2*b-3)*(G/2) & V < S + (2*b-1)*(G/2)) ,1);
                    freqCount(c,d+1,b+1) = binCount;
                    P = freqCount(c,d+1,b+1)/(sampleSize*G);
                end
                formatSpec = 'Class %d, attribute %d, bin %d, P(bin | class) = %.2f\n';
                fprintf(formatSpec,uniqueClasses(c),d,b,P);
            end
        end
    end
    %classification
    for i=1:testSize
        X = testFile(i,:);
        prob = zeros(NoOfClasses,1);
        num = zeros(100,1);
        denom = 0;
        for c=1:NoOfClasses
            V_prob = 1.0;
            for d=0:colSize-2
                sig = bound(c,d+1,1);
                L = bound(c,d+1,2);
                G = (L-sig)/(N-3);
                G = max(G,0.0001);
                if X(1,d+1) < sig-(G/2)
                    b = 0;
                elseif X(1,d+1) >= L+(G/2)
                    b = N-1;
                else
                    b = floor(( X(1,d+1) - (sig-G/2) )/G)+1;
                    b = max(b,0);
                    b = min(b,N-1);
                end
                b = 1;
                tmp = 0;
                while ((X(1, d+1) >= sig-(G/2) + tmp) && (b < N))
                    b = b + 1;
                    tmp = tmp + G;
                end
                V_prob = V_prob*(freqCount(c, d+1, b)/( bound(c,d+1,3)*G) );
            end
            V_prob = V_prob*(countClass(c)/rowSize); %Prob from training dataset
            prob(c,1) = V_prob;
            num(uniqueClasses(c,1)) = V_prob;
            denom = denom + V_prob;
        end
        mi = max(prob);
        classSearch = uniqueClasses(find(prob == mi),1);
        cells = find(classSearch == X(1,colSize));
        accuracy = 0;
        if size(cells,1) > 0
            accuracy = (1/size(classSearch,1));
            overall_accuracy = overall_accuracy + accuracy;
        end
        sz = size(classSearch,1);
        predicted_class = classSearch( randi([1, sz]),1);
        if denom == 0
            P = 1/NoOfClasses; %distribute the probability
        else
            P = num(predicted_class)/denom;
        end
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n', i-1,predicted_class , P, testFile(i,colSize), accuracy);
    end
end

%GAUSSIAN

if strcmp(type, 'gaussians')
    %Training
    for c = 1:NoOfClasses
        index = find(trainFile(:,colSize)==uniqueClasses(c));
        countClass(c) = size(index,1);
        for d = 0:colSize-2
            V = trainFile(index,d+1);
            mu(c,d+1) = mean(V);
            sig(c,d+1) = std(V);
            if sig(c,d+1) < 0.01
                sig(c,d+1) = 0.01;
            end
            
            formatSpec = 'Class %d, attribute %d, mean = %.2f, std = %.2f\n';
            fprintf(formatSpec,uniqueClasses(c),d,mu(c,d+1),sig(c,d+1));
        end
    end
    
    %classification
    for i=1:testSize
        X = testFile(i,:);
        prob = zeros(NoOfClasses,1);
        num = zeros(100,1);
        denom = 0;
        for c=1:NoOfClasses
            V_prob = 1.0;
            for d=0:colSize-2
                gauss = normpdf(X(1,d+1), mu(c,d+1,1), sig(c,d+1,1));
                V_prob = V_prob*( gauss );
            end
            V_prob = V_prob*(countClass(c)/rowSize); %Prob from training dataset
            prob(c,1) = V_prob;
            num(uniqueClasses(c,1)) = V_prob;
            denom = denom + V_prob;
        end
        mi = max(prob);
        classSearch = uniqueClasses(find(prob == mi),1);
        cells = find(classSearch == X(1,colSize));
        accuracy = 0;
        if size(cells,1) > 0
            accuracy = (1/size(classSearch,1));
            overall_accuracy = overall_accuracy + accuracy;
        end
        sz = size(classSearch,1);
        predicted_class = classSearch( randi([1, sz]),1);
        if denom == 0
            P = 1/NoOfClasses;  %distribute the probability
        else
            P = num(predicted_class)/denom;
        end
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n', i-1,predicted_class , P, testFile(i,colSize), accuracy);
    end
end

%Mixtures of Gaussians
if strcmp(type, 'mixtures')
    %Training
    N = str2num(varargin{4});
    M = zeros(NoOfClasses,colSize-1,N);
    SIG = ones(NoOfClasses,colSize-1,N);
    W = ones(NoOfClasses,colSize-1,N)/N;
    for c = 1:NoOfClasses
        index = find(trainFile(:,colSize)==uniqueClasses(c));
        for d = 0:colSize - 2
            V = trainFile(index,d+1);
            L = max(V);
            S = min(V);
            G = (L-S)/N;
            G - max(G,0.0001);
            for gauss = 1:N
                M(c,d+1,gauss) = S + (gauss-1)*G + G/2;
            end
            %EM ALGO
            Ni = zeros(size(V,1),N);
            Pij = zeros(size(V,1),N);
            sz = size(V,1);
            for iter = 1:50
                for j = 1:sz
                    denom = 0;
                    for i = 1:N
                        Ni(j,i) = normpdf(V(j,1),M(c,d+1,i),SIG(c,d+1,i));
                        Pij(j,i) = Ni(j,i)*W(c,d+1,i);
                        denom = denom + Pij(j,i);
                    end
                    Pij(j,:) = Pij(j,:)/denom;
                end
                
                for i = 1:N
                    M(c,d+1,i) = ( sum( Pij(:,i).*(V(:,1)) ) )/(sum(Pij(:,i))); %Mean Update
                    VAR = max((sum( Pij(:,i).*( (V(:,1) - M(c,d+1,i)).^2 ) ) )/( sum(Pij(:,i))),0.0001); %Variance Update
                    SIG(c,d+1,i) = sqrt( VAR );
                    W(c,d+1,i) = sum(Pij(:,i))/ (sum( Pij(:))); %Weight Update
                end
                
                
            end
            
            for i = 1: N
                formatSpec = 'Class %d, attribute %d, Gaussian %d, mean = %.2f, std = %.2f\n';
                fprintf(formatSpec,uniqueClasses(c),d,i-1,M(c,d+1,i),SIG(c,d+1,i));
            end
            
        end
    end
    
    %classification
    for i=1:testSize
        V = testFile(i,:);
        prob = zeros(NoOfClasses,1);
        num = zeros(100,1);
        denom = 0;
        for c=1:NoOfClasses
            V_prob = 1.0;
            for d=0:colSize-2
                gauss = 0;
                for i=1:N
                    gauss = gauss + W(c,d+1,i,1)*normpdf(V(1,d+1), M(c,d+1,i,1), SIG(c,d+1,i,1));
                end
                V_prob = V_prob*( gauss );
            end
            V_prob = V_prob*(countClass(c)/rowSize); %Prob from training dataset
            prob(c,1) = V_prob;
            num(uniqueClasses(c,1)) = V_prob;
            denom = denom + V_prob;
        end
        mi = max(prob);
        classSearch = uniqueClasses(find(prob == mi),1);
        cells = find(classSearch == V(1,colSize));
        accuracy = 0;
        if size(cells,1) > 0
            accuracy = (1/size(classSearch,1));
            overall_accuracy = overall_accuracy + accuracy;
        end
        sz = size(classSearch,1);
        predicted_class = classSearch( randi([1, sz]),1);
        if denom == 0
            P = 1/NoOfClasses; %distribute the probability
            
        else
            P = num(predicted_class)/denom;
        end
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n', i-1,predicted_class , P, testFile(i,colSize), accuracy);
    end
end
fprintf('classification accuracy=%6.4f\n',overall_accuracy/testSize);