%Author Ashiq Imran
function [ output_args ] = dtw_classify(varargin )
    
    training = varargin{1};
    testing = varargin{2};
    
    [trainObj,trainData, trainSize] = ParseInput(fopen(training));
    [testObj,testData, testSize] = ParseInput(fopen(testing));
    overall_accuracy = 0;
    for x=1:testSize
        predicted_class = 0;
        min_dist = Inf;
        for y=1:trainSize
            A = testData(x).attr;
            B = trainData(y).attr;
            dist = DTW(A,B);   %DTW function
            if (dist < min_dist)
                min_dist = dist;
                predicted_class = trainObj(y,2);
            end
        end
        if predicted_class == testObj(x,2)
           overall_accuracy = overall_accuracy+1;
           accuracy = 1;
        else
           accuracy = 0;
        end
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f, distance = %.2f\n', x, predicted_class, testObj(x,2), accuracy, min_dist);
    end
    fprintf('classification accuracy=%6.4f\n',overall_accuracy/testSize); 
    fclose('all');
end


function [cls,data, sz] = ParseInput(file)
    tline = fgets(file);
    tline = fgets(file);
    i = 1;
    data(1:4000) = objClass; 
    cls = [];
    while ischar(tline)
        if (strcmp(tline(1), 'o')) > 0
            pos = find(tline == ':');
            objID = str2double( tline(pos(1)+2 : length(tline)) );
        end
        tline = fgets(file);
        if (strcmp(tline(1), 'c')) > 0
            pos = find(tline == ':');
            classID = str2double( tline(pos(1)+2 : length(tline)) );
        end
        cls = [cls; objID,classID];
        tline = fgets(file);
        tline = fgets(file);
        tline = fgets(file);
        if (strcmp(tline(1), 'd')) > 0
            tline = fgets(file);
            tmp = [];
            while ischar(tline) && size(strfind(tline, '--'),2) == 0
                A = sscanf(tline,'%f');
                tmp = [tmp;A(1,1), A(2,1)];
                tline = fgets(file);
            end
            size(tmp);
            instance = objClass;
            instance.attr = tmp;
            data(i) = instance;
        end
        i = i+1;
        if ischar(tline) == false
            break;
        end
        tline = fgets(file);
    end
    sz = i-1;
    fclose(file);
end
%%%%%%%%%%%%%%%%%%%%%% DTW %%%%%%%%%%%%%%%%%%%%%%%%%%
function dist = DTW(A, B)
    N = size(A,1);
    M = size(B,1);
    arr = zeros(N,M);
    arr(1,1) = sqrt(((A(1,1) - B(1,1))^2) + ((A(1,2) - B(1,2))^2));
    for i=2:N
      arr(i,1) = arr(i-1,1)+ sqrt(((A(i,1) - B(1,1))^2) + ((A(i,2) - B(1,2))^2)); 
    end
    for i=2:M
      arr(1,i) = arr(1,i-1)+ sqrt(((A(1,1) - B(i,1))^2) + ((A(1,2) - B(i,2))^2)); 
    end
    for i=2:N
      for j=2:M
          MIN = min(arr(i-1,j), arr(i,j-1));
          MIN = min(MIN, arr(i-1,j-1));
          arr(i,j) = MIN + sqrt(((A(i,1) - B(j,1))^2) + ((A(i,2) - B(j,2))^2));
      end
    end
    dist = arr(N,M);
end