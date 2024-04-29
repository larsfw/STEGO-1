clc
clear
close all

labelledPath = 'D:\waglar\PotsdamMATLAB\potsdamtraining_six_channels\labelled.txt';
trainPath = 'D:\waglar\PotsdamMATLAB\potsdamtraining_six_channels\labelled_train.txt';
testPath = 'D:\waglar\PotsdamMATLAB\potsdamtraining_six_channels\labelled_test.txt';

fid = fopen(labelledPath, 'r');
labelled = textscan(fid, '%d');
fclose(fid);
labelled = labelled{1};

train = labelled(1:4545);
test = labelled(4546:5400);

writeIndicesToFile(train, trainPath);
writeIndicesToFile(test, testPath);


function writeIndicesToFile(indices, filePath)

    fileID = fopen(filePath, 'w');

    if fileID == -1
        error('Error creating the file.');
    end

    fprintf(fileID, '%d\n', indices);

    fclose(fileID);

    fprintf('Indices written to: %s\n', filePath);
end