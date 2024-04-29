clc
clear
close all

%%% load mat files from potsdam raw %%%

matFolderPath = 'D:\waglar\PotsdamMATLAB\potsdamraw';
matFiles = dir(fullfile(matFolderPath, '*.mat'));
patches = cell(1, numel(matFiles));

% Loop through each .mat file and extract each img array
for i = 1:numel(matFiles)
    data = load(fullfile(matFolderPath, matFiles(i).name));
    patches{i} = data.img;
end

%%% calculate NDVI and replace fourth channel with it %%%

% Loop through each img array in imgArrays
for i = 1:numel(patches)
    % NDVI = (NIR - R) / (NIR + R)
    NDVI = (double(patches{i}(:,:,4))-double(patches{i}(:,:,1)))./(double(patches{i}(:,:,4))+double(patches{i}(:,:,1)));
    patches{i}(:,:,4) = uint8(127 * (NDVI+1));
end

%%% load PMAP and NDSM %%%

tifFolderPath = 'D:\waglar\PotsdamMATLAB\NDVI_PMAP_NDSM_10cm';
tifFiles = dir(fullfile(tifFolderPath, '*.tif'));
tifPatches = zeros(200, 200, 2, 225 * numel(tifFiles));

for fileIdx = 1:numel(tifFiles)
    tifData = imread(fullfile(tifFolderPath, tifFiles(fileIdx).name));
    % Check if the size of the TIFF file matches the expected size (3000x3000x3)
    if size(tifData, 1) == 3000 && size(tifData, 2) == 3000 && size(tifData, 3) == 3
        % Extract patches from the last two channels
        for i = 1:15
            for j = 1:15
                startRow = (i - 1) * 200 + 1;
                endRow = startRow + 199;
                startCol = (j - 1) * 200 + 1;
                endCol = startCol + 199;

                patch = tifData(startRow:endRow, startCol:endCol, 2:3);
                index = ((fileIdx - 1) * 15 + i - 1) * 15 + j;

                tifPatches(:, :, :, index) = patch;
            end
        end
    else
        fprintf('Skipped file %s because it does not have the expected size.\n', tifFiles(fileIdx).name);
    end
end

% Now tifPatches contains the 200x200x2 patches from the last two channels of each TIFF file

for i = 1:size(patches, 2)
    % Concatenate the two tifPatch channels to the four existing channels in patches
    patches{i} = cat(3, patches{i}, tifPatches(:,:,:,i));
end

fprintf('Size of first patch: %s\n', mat2str(size(patches{1})));

%%% save mat files %%%

outputFolder = 'D:\waglar\PotsdamMATLAB\potsdamraw_NDVI_PMAP_NDSM';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
numSaves = 0;
for i = 1:numel(patches)
    img = patches{i};
    fileName = matFiles(i).name;
    filePath = fullfile(outputFolder, fileName);
    try
        save(filePath, 'img');
        numSaves = numSaves +1;
    catch
        disp(['Error saving entry ' num2str(i) ' to ' filePath]);
    end
end

disp(['Successfully saved ' num2str(numSaves) ' .mat files to ' outputFolder])
