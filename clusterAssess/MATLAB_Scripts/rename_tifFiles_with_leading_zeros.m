clc
clear
close all

% Path to the folder containing the files
folderPath = 'D:\waglar\PotsdamMATLAB\NDVI_PMAP_NDSM_10cm';

% Get a list of all .mat files in the folder
files = dir(fullfile(folderPath, '*.tif'));

% Preallocate the renameLog struct
renameLog = struct('oldName', cell(1, numel(files)), 'newName', cell(1, numel(files)));

% Loop through each file and rename it with leading zeros
for i = 1:numel(files)
    oldName = fullfile(folderPath, files(i).name);
    
    % Extract the three numeric parts of the filename
    [~, fileName, fileExt] = fileparts(files(i).name);
    numericParts = regexp(fileName, '\d+', 'match');
    
    % Add leading zeros to all numeric parts except the last one
    formattedNumericParts = cellfun(@(x) sprintf('%02d_', str2double(x)), numericParts(1:end-1), 'UniformOutput', false);
    
    % Append the last numeric part without an underscore
    formattedNumericParts{end+1} = sprintf('%02d', str2double(numericParts{end})); 
    
    % Concatenate the parts into a single string
    formattedNumericString = strcat([formattedNumericParts{:} '_NDVI_PMAP_NDSM']);
    
    % Create the new filename with underscores between numeric parts
    newName = fullfile(folderPath, [formattedNumericString fileExt]);
   
    % Check if the new name is different from the old name before moving
    if ~isequal(oldName, newName)
        % Rename the file only if the names are different
        movefile(oldName, newName);
        
        % Store old and new filenames in the preallocated struct
        renameLog(i).oldName = oldName;
        renameLog(i).newName = newName;
    else
        % If the names are the same, just store them in the log without moving
        renameLog(i).oldName = oldName;
        renameLog(i).newName = newName;
    end
end