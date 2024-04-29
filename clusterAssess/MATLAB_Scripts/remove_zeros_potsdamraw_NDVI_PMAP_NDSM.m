clc
clear
close all

% Specify the folder containing the .mat files
matFolderPath = 'D:\waglar\PotsdamMATLAB\potsdamraw_NDVI_PMAP_NDSM';
outputFolder = 'D:\waglar\PotsdamMATLAB\potsdamraw_NDVI_PMAP_NDSM\processed\imgs';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(matFolderPath, '*.mat'));

% Loop through each .mat file
for i = 1:numel(matFiles)
    % Extract the file name
    filename = matFiles(i).name;
    
    % Extract the indices from the filename using regular expressions
    indices = regexp(filename, '\d+', 'match');
    
    % Remove leading zeros from each index
    indices_without_zeros = arrayfun(@(x) num2str(str2double(x)), indices, 'UniformOutput', false);
    
    % Construct the new filename without leading zeros
    new_filename = [strjoin(indices_without_zeros, '_'), '.mat'];
    
    % Rename the file
    movefile(fullfile(matFolderPath, filename), fullfile(outputFolder, new_filename));
end