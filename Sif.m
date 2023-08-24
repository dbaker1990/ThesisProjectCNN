%unzip('train.zip');

fileList = dir('.\train\No Impairment\*.jpg');
for idx = 1:length(fileList)
    imG = imread(['.\train\No Impairment\' fileList(idx).name]);
    imRGB = zeros([size(imG, 1), size(imG, 2), 3], 'uint8');
    imRGB(:,:,1) = imG;
    imRGB(:,:,2) = imG;
    imRGB(:,:,3) = imG;
    imwrite(imRGB, ['.\train_tiff\No Impairment\' fileList(idx).name]);
end

fileList2 = dir('.\train\Moderate Impairment\*.jpg');
for idx = 1:length(fileList2)
    imG = imread(['.\train\Moderate Impairment\' fileList2(idx).name]);
    imRGB = zeros([size(imG, 1), size(imG, 2), 3], 'uint8');
    imRGB(:,:,1) = imG;
    imRGB(:,:,2) = imG;
    imRGB(:,:,3) = imG;
    imwrite(imRGB, ['.\train_tiff\Moderate Impairment\' fileList2(idx).name]);
end

fileList3 = dir('.\train\Mild Impairment\*.jpg');
for idx = 1:length(fileList3)
    imG = imread(['.\train\Mild Impairment\' fileList3(idx).name]);
    imRGB = zeros([size(imG, 1), size(imG, 2), 3], 'uint8');
    imRGB(:,:,1) = imG;
    imRGB(:,:,2) = imG;
    imRGB(:,:,3) = imG;
    imwrite(imRGB, ['.\train_tiff\Mild Impairment\' fileList3(idx).name]);
end

fileList4 = dir('.\train\Very Mild Impairment\*.jpg');
for idx = 1:length(fileList4)
    imG = imread(['.\train\Very Mild Impairment\' fileList4(idx).name]);
    imRGB = zeros([size(imG, 1), size(imG, 2), 3], 'uint8');
    imRGB(:,:,1) = imG;
    imRGB(:,:,2) = imG;
    imRGB(:,:,3) = imG;
    imwrite(imRGB, ['.\train_tiff\Very Mild Impairment\' fileList4(idx).name]);
end
imds = imageDatastore('train_tiff', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


%Divide the data into training and validation data sets. Use 70% of the images for training and 30% for validation. 
% splitEachLabel splits the images datastore into two new datastores.
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,0.2,'randomized');

%imread-read image file to variable
%

%This very small data set now contains 55 training images and 20 validation images. Display some sample images.
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,1000);


layers = [
    imageInputLayer([128 128 3],"Name","imageinput")
    convolution2dLayer([3 3],128,"Name","conv","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same","Stride",[2 2])
    maxPooling2dLayer([5 5],"Name","maxpool","Padding","same","Stride",[2 2])
    eluLayer(1,"Name","elu")
    convolution2dLayer([3 3],32,"Name","conv_4","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_5","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_6","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_7","Padding","same","Stride",[2 2])
    maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same","Stride",[2 2])
    eluLayer(1,"Name","elu_1")
    convolution2dLayer([3 3],32,"Name","conv_8","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_9","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_10","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_11","Padding","same","Stride",[2 2])
    maxPooling2dLayer([5 5],"Name","maxpool_2","Padding","same","Stride",[2 2])
    eluLayer(1,"Name","elu_2")
    convolution2dLayer([3 3],8,"Name","conv_12","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],4,"Name","conv_13","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],8,"Name","conv_14","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],4,"Name","conv_15","Padding","same","Stride",[2 2])
    maxPooling2dLayer([5 5],"Name","maxpool_3","Padding","same","Stride",[2 2])
    eluLayer(1,"Name","elu_3")
    convolution2dLayer([3 3],8,"Name","conv_16","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_17","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_18","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_19","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_20","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],4,"Name","conv_21","Padding","same","Stride",[2 2])
    maxPooling2dLayer([5 5],"Name","maxpool_4","Padding","same","Stride",[2 2])
    eluLayer(1,"Name","elu_4")
    layerNormalizationLayer("Name","layernorm")
    reluLayer("Name","relu")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(4,"Name","fc")
    classificationLayer("Name","classoutput")];
layer = fullyConnectedLayer(4);

options = trainingOptions("sgdm","InitialLearnRate",1e-4,"MaxEpochs",20,"Shuffle","every-epoch","ValidationData",imdsValidation,"ValidationFrequency",30,"Verbose",false,"Plots","training-progress");

net = trainNetwork(imds,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);