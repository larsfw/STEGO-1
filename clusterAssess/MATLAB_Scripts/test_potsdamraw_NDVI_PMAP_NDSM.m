clc
clear
close all

matFolderPath = 'D:\waglar\PotsdamMATLAB\potsdamraw_NDVI_PMAP_NDSM';
matFiles = dir(fullfile(matFolderPath, '*.mat'));
patches = cell(1, numel(matFiles));

for i = 1:numel(matFiles)
    data = load(fullfile(matFolderPath, matFiles(i).name));
    patches{i} = data.img;
end

numPatchesToDisplay = 6;
randomIndices = randperm(numel(patches), numPatchesToDisplay);
figure;

for i = 1:numPatchesToDisplay
    patch = patches{randomIndices(i)};
    
    img1 = patch(:, :, 1:3);
    subplot(2, numPatchesToDisplay, i);
    imshow(img1);
    title([matFiles(randomIndices(i)).name ' | R-G-B']);
    set(get(gca, 'Title'), 'String', strrep(get(get(gca, 'Title'), 'String'), '_', '\_'));

    img2 = patch(:, :, 4:6);
    subplot(2, numPatchesToDisplay, i + numPatchesToDisplay);
    imshow(img2);
    title([matFiles(randomIndices(i)).name ' | NDVI-PMAP-NDSM']);
    set(get(gca, 'Title'), 'String', strrep(get(get(gca, 'Title'), 'String'), '_', '\_'));
end