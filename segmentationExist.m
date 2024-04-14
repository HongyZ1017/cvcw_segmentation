clc;
clear all;
close all;

imageDir = fullfile('data_for_moodle/images_256/')
labelDir = fullfile('data_for_moodle/labels_256/')
SortlabelDir = fullfile('labels/')

%convert all label image into 2 classes: background and flower
% for i = 1:length(f)
%     [labelImage, map]= imread(fullfile(labelDir, f(i).name));
%     labelImage(labelImage ~= 1) = 3
%     imwrite(labelImage, map, fullfile(SortlabelDir, f(i).name));
% end


classNames = ["background", "flower"]
pixelLabel = [3 1]
imds = imageDatastore(imageDir)
lbds = pixelLabelDatastore(SortlabelDir, classNames, pixelLabel);


%select image and related index
[~, filename, ~] = cellfun(@fileparts,imds.Files,'UniformOutput',false)
[~, labelname, ~] = cellfun(@fileparts,lbds.Files,'UniformOutput',false)
[commonFiles, iim, ilb] = intersect(filename,labelname)


image_size = [256 256 3]
numClasses = 2
lgraph = unetLayers(image_size, numClasses)

test_rate_set = [0.2, 0.15, 0.25]
for t = 1: length(test_rate_set)
    test_rate = test_rate_set(t)
    cvpTest = cvpartition(length(commonFiles), 'HoldOut', test_rate)
    testIdx = test(cvpTest)
    trainValIdx = training(cvpTest)
    
    testimds = subset(imds, iim(testIdx))
    testlbds = subset(lbds, ilb(testIdx))
    % testset = combine(testimds, testlbds)
    testset = pixelLabelImageDatastore(testimds, testlbds)
    
    %tuning hyper parameter
    n = [5, 6, 7]
    for i = 1: length(n)
        
        cvpCross = cvpartition(sum(trainValIdx), 'KFold', n(i))
    
        for j = 1:n(i)
            trainIdx = training(cvpCross, j)
            valIdx = test(cvpCross, j)
            
            trainimds = subset(imds, iim(trainIdx))
            valimds = subset(imds, iim(valIdx))
            trainldbs = subset(lbds, ilb(trainIdx))
            valildbs =subset(lbds, ilb(valIdx))
            
            trainset = pixelLabelImageDatastore(trainimds, trainldbs)
            valiset = pixelLabelImageDatastore(valimds, valildbs)
            % trainset = combine(trainimds, trainldbs) 
            % valiset = combine(valimds, valildbs)
    
            %get frequency of classes
            tbl = countEachLabel(trainldbs)
            totalNumOfPixels = sum(tbl.PixelCount);
            frequency = tbl.PixelCount / totalNumOfPixels
            classWeights = 1./frequency
            
            
            
            opts = trainingOptions('sgdm', ...
                'InitialLearnRate',1e-3, ...
                'MaxEpochs',200, ...
                'MiniBatchSize',32, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', valiset, ...
                'ValidationFrequency', 30, ...
                'Verbose', true, ...
                'Plots', 'training-progress', ...
                'ValidationPatience', 10)


            [net, trainInfo] = trainNetwork(trainset, lgraph, opts)
            

            %Do segmentation, save output images to disk
           
            break

        end
        break
    end
    break
end
