clc
clear
close all

imgsInputFolder = 'D:\waglar\PotsdamMATLAB\imgs_input';
gtInputFolder = 'D:\waglar\PotsdamMATLAB\gt_input';
imgsOutputFolder = 'D:\waglar\PotsdamMATLAB\potsdamtraining_six_channels\imgs';
gtOutputFolder = 'D:\waglar\PotsdamMATLAB\potsdamtraining_six_channels\gt';

if ~exist(imgsOutputFolder, 'dir')
    mkdir(imgsOutputFolder);
end

if ~exist(gtOutputFolder, 'dir')
    mkdir(gtOutputFolder);
end

imgsMatFiles = dir(fullfile(imgsInputFolder, '*.mat'));
gtMatFiles = dir(fullfile(gtInputFolder, '*.mat'));
uniqueRandomNumbers = randperm(length(imgsMatFiles));
labelled = [];
unlabelled = [];

% Imagenumbers without ground truth
excludeNumbers = [3,4,8,9,13,14,15,19,20,21,25,26,27,34];

for i = 1:length(imgsMatFiles)
    randomIndex = uniqueRandomNumbers(i);
    fileNumber = str2double(imgsMatFiles(i).name(1:2));

    currentImg = load(fullfile(imgsInputFolder, imgsMatFiles(i).name));
    img = currentImg.img;
    newFileName = sprintf('%d.mat', randomIndex);

    if ~any(fileNumber == excludeNumbers)
        currentGt = load(fullfile(gtInputFolder, imgsMatFiles(i).name));
        gt = currentGt.gt;
        save(fullfile(gtOutputFolder, newFileName), 'gt');
        labelled = [labelled, randomIndex]; %#ok<AGROW> 
    end
    
    if ~ismember(randomIndex, labelled)
        unlabelled = [unlabelled, randomIndex]; %#ok<AGROW> 
    end
        
    save(fullfile(imgsOutputFolder, newFileName), 'img');
    fprintf('Processed file %d of %d\n', i, length(imgsMatFiles));
end

writeIndicesToFile(labelled, 'D:\waglar\PotsdamMATLAB\potsdamtraining_six_channels\labelled.txt');
writeIndicesToFile(unlabelled, 'D:\waglar\PotsdamMATLAB\potsdamtraining_six_channels\unlabelled_train.txt');

function writeIndicesToFile(indices, filePath)

    fileID = fopen(filePath, 'w');

    if fileID == -1
        error('Error creating the file.');
    end

    fprintf(fileID, '%d\n', indices);

    fclose(fileID);

    fprintf('Indices written to: %s\n', filePath);
end