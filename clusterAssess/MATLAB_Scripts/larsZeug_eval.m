%cd tests
%myPath = 'C:\Daten\Temp_Data\ReiseHannover\2023 Zwischenbericht 3\potsdam\';
%myPath = 'C:\Daten\Temp_Data\potsdam\';
% myPath = 'D:\waglar\STEGO\results\predictions\potsdam\';
myPath = 'D:\waglar\results\predictions\potsdam_fusion0_date_Apr25_18-50-17\';
gtPath = 'D:\waglar\Potsdam_Labels\5_Labels_all\';

firstRun = 0;


load ('D:\waglar\results\predictions\potsdamO', 'myPotsdam', 'myPotsdamO');


amiAll = zeros(numel(myPotsdamO),1); ariAll = amiAll; pixAll = amiAll; 
confusionAll = cell(numel(myPotsdamO), 1);
%% 
for u = 0:numel(myPotsdamO)-1
   i1 = imread ([myPath 'cluster\' num2str(u) '_crf.png']);
   i2 = imread ([myPath 'cluster\' num2str(u) '_no_crf.png']);
   i3 = imread ([myPath 'img\' num2str(u) '.png']);
   if firstRun
       i4 = imread ([myPath 'label\' num2str(u) '.png']); 
   else
       i4 = imread([gtPath, myPotsdamO(u+1).name]);
       i4 = imresize(i4, 0.8, 'nearest');
       if ~isempty(find((i4 < 50 & i4 > 0) | (i4 > 200 & i4 < 255), 1))
          disp(['Patch ' num2str(u) ' kaputtinterpoliert!']);
          i4(i4 < 50) = 0;
          i4(i4 > 200) = 255; 
       end
   end
   %display_input_gr_axkt_18([], [], 2, 2, i1, i2, i3, i4);
   
   [H, W, f] = size(i1);
   [a1, ~, c1] = unique(reshape(i1, H*W, f), 'rows');
   [a4, ~, c4] = unique(reshape(i4, H*W, f), 'rows');
   
   amiIndex =ami(c1, c4);
   arIndex = rand_index(c1, c4, 'adjusted');
   
   amiAll(u+1) = amiIndex;
   ariAll(u+1) = arIndex;
   pixAll(u+1) = H*W;
   confusionAll{u+1}=confusionmat(c1, c4);
   
end

%
save D:\waglar\STEGO\results\predictions\result_potsdam_fusion0_date_Apr25_18-50-17 *All*