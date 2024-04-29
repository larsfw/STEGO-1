clc
clear
close all


im1 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_0.mat');
im2 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_1.mat');
im3 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_2.mat');
im4 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_3.mat');
im5 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_4.mat');
im6 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_5.mat');
im7 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_6.mat');
im8 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_7.mat');
im9 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_8.mat');
im10 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_9.mat');
im11 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_10.mat');
im12 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_11.mat');
im13 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_12.mat');
im14 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_13.mat');
im15 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_14.mat');

display_input_gr_axkt_18([],[],2,4,jet(128),im1.img(:,:,1:3),im2.img(:,:,1:3),im3.img(:,:,1:3),im4.img(:,:,1:3),im5.img(:,:,1:3),im6.img(:,:,1:3),im7.img(:,:,1:3),im8.img(:,:,1:3));

% load the tif files
raw0 = imread('D:\waglar\pytorch_datasets\potsdamraw\Potsdam\4_Ortho_RGBIR\top_potsdam_2_10_RGBIR.tif');
raw34 = imread('D:\waglar\pytorch_datasets\potsdamraw\Potsdam\4_Ortho_RGBIR\top_potsdam_7_13_RGBIR.tif');

% take the first row of the first image
raw0FirstRow = raw0(1:400,:,:);

% extract the first three channels (RGB conversion)
rgb_image = raw0FirstRow(:,:,1:3);

% display the RGB image
imshow(rgb_image);

% Use array comprehension to extract arrays from each struct
imageArrayCell = arrayfun(@(s) s.img, [im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12, im13, im14, im15], 'UniformOutput', false);

% Concatenate the arrays along the second dimension
concatenatedArray = cat(2, imageArrayCell{:});

% Display the size of the concatenated array
disp(size(concatenatedArray));

% extract the first three channels (RGB conversion)
rgb_image1 = concatenatedArray(:,:,1:3);

% display the RGB image
imshow(rgb_image1);

% Define the target size
targetSize = [200, 3000];

% Resize the array
resizedRaw0FirstRow = imresize(raw0FirstRow, targetSize,"box");

% extract the first three channels (RGB conversion)
rgb_image2 = resizedRaw0FirstRow(:,:,1:3);

% display the RGB image
imshow(rgb_image2);

if isequal(concatenatedArray, resizedRaw0FirstRow)
    disp('Arrays are the same.');
else
    disp('Arrays are different.');
end

differences = concatenatedArray ~= resizedRaw0FirstRow;
same = concatenatedArray == resizedRaw0FirstRow;

% Count the number of differing elements
numDifferences = nnz(differences);
numSame = nnz(same);

% Display the result
disp([num2str(round((numSame/(numSame + numDifferences)*100),2)) ' % of entries are the exact same.']);




