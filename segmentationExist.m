clc;
clear all;
close all;

rng('default');
rng(21);

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

global best_Accuracy bestNet bestinfo
best_Accuracy = 0
bestNet = []

results = table('Size', [0 6], ...
    'VariableTypes', {'int32','int32', 'int32', 'double', 'double', 'cell'}, ...
    'VariableNames', {'N_vali','maxEpoch', 'miniBatchSize', 'initailLearningRate', 'valiAccuarcy', 'modelName'});

%select image and related index
[~, filename, ~] = cellfun(@fileparts,imds.Files,'UniformOutput',false)
[~, labelname, ~] = cellfun(@fileparts,lbds.Files,'UniformOutput',false)
[commonFiles, iim, ilb] = intersect(filename,labelname)


image_size = [256 256 3]
numClasses = 2
% use exist network
lgraph = unetLayers(image_size, numClasses)


test_rate = 0.2
cvpTest = cvpartition(length(commonFiles), 'HoldOut', test_rate)
testIdx = test(cvpTest)
trainValIdx = training(cvpTest)

testimds = subset(imds, iim(testIdx))
testlbds = subset(lbds, ilb(testIdx))
% testset = combine(testimds, testlbds)
testset = pixelLabelImageDatastore(testimds, testlbds)

%tuning hyper parameter
n = 4
cvpCross = cvpartition(sum(trainValIdx), 'KFold', n)

%training process
for j = 1: 4
    N_vali = j

    trainIdx = training(cvpCross, j)
    valIdx = test(cvpCross, j)

    trainimds = subset(imds, iim(trainIdx))
    valimds = subset(imds, iim(valIdx))
    trainldbs = subset(lbds, ilb(trainIdx))
    valildbs =subset(lbds, ilb(valIdx))

    trainset = pixelLabelImageDatastore(trainimds, trainldbs)
    valiset = pixelLabelImageDatastore(valimds, valildbs)

    %get frequency of classes
    tbl = countEachLabel(trainldbs)
    totalNumOfPixels = sum(tbl.PixelCount);
    frequency = tbl.PixelCount / totalNumOfPixels
    classWeights = 1./frequency

    epoch_set = 3
    minibatch = 32
    inital_learn_rates = 0.001


    modelName = sprintf('%dvali_%depoch_%dbatch_%.3flearnRate.mat', j, epoch_set, minibatch, inital_learn_rates)

    opts = trainingOptions('adam', ...
        'InitialLearnRate',inital_learn_rates, ...
        'MaxEpochs',epoch_set, ...
        'MiniBatchSize',minibatch, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', valiset, ...
        'ValidationFrequency', 5, ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ValidationPatience', 5)


    [net, info] = trainNetwork(trainset, lgraph, opts)

    if info.FinalValidationAccuracy > best_Accuracy
        best_Accuracy = info.FinalValidationAccuracy
        bestNet = net
        bestinfo = info
        save("bestNetExist.mat", 'net')
    end

    currentmeanPixel_Accuracy = max([info.ValidationAccuracy])
    newResult = table(j,epoch_set, minibatch,inital_learn_rates, currentmeanPixel_Accuracy, {modelName},...
        'VariableNames', {'N_vali','maxEpoch', 'miniBatchSize', 'initailLearningRate', 'valiAccuarcy', 'modelName'})


    % Collect and save training results
    results = [results; newResult]


end

%use best model for test_set segmentation
load('bestNetExist.mat');
predictedLabels = semanticseg(testimds, net);
metrics = evaluateSemanticSegmentation(predictedLabels, testlbds);

% Display the metrics
disp(metrics.ConfusionMatrix);
disp(['Global Accuracy: ', num2str(metrics.DataSetMetrics.GlobalAccuracy)]);

%find the best and worst image
perImageResults = metrics.ImageMetrics
accuracies = perImageResults.MeanAccuracy
[~, bestIdx] = max(accuracies)
[~, worstIdx] = min(accuracies)

disp(['Best Image Accuracy: ', num2str(accuracies(bestIdx))])
disp(['Worst Image Accuracy: ', num2str(accuracies(worstIdx))])
bestlbds = readimage(testlbds, bestIdx)
bestprelbds = readimage(predictedLabels, bestIdx)
worstlbds = readimage(testlbds, worstIdx)
worstprelbds = readimage(predictedLabels, worstIdx)

[~, map]= imread('data_for_moodle\labels_256\image_0004.png')

bestlbds = ind2rgb(bestlbds, map)
bestprelbds = ind2rgb(bestprelbds, map)
worstlbds = ind2rgb(worstlbds, map)
worstprelbds = ind2rgb(worstprelbds, map)

f =figure;
subplot(2,2,1)
imshow(bestlbds)
subplot(2,2,2)
imshow(bestprelbds)
title(['Best Predicted Image, Accuracy: ', num2str(accuracies(bestIdx))])

subplot(2,2,3)
imshow(worstlbds)
subplot(2,2,4)
imshow(worstprelbds)
title(['Worst Predicted Image, Accuracy: ', num2str(accuracies(worstIdx))])
saveas(f, 'exist_best_worst.jpg')

f = figure;
subplot(1,2,1)
imshow(bestlbds);
hold on;
h = imshow(bestprelbds);
set(h, 'AlphaData', 0.5);
title('Best Prediction Overlay on True Labels');
hold off;
subplot(1,2,2)
imshow(worstlbds);
hold on;
h = imshow(worstprelbds);
set(h, 'AlphaData', 0.5);
title('Worst Prediction Overlay on True Labels');
hold off;
saveas(f, 'exist_overlay.jpg')

%convert generate ind image to rgb image
outputDir = 'segmented_images_exist';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


[~, map]= imread('labels\image_0004.png');
idx = 1;
reset(predictedLabels); % reset pin
while hasdata(predictedLabels)

    [cellArray, info] = read(predictedLabels);  % read cell
    labelImage = cellArray{1}; 

    rgbImage = ind2rgb(labelImage, map);

%  save image
    [~, fileName, ~] = fileparts(info.Filename);
    outputFile = fullfile(outputDir, [fileName, '.png']);
    imwrite(rgbImage, outputFile);
    idx = idx + 1;
end