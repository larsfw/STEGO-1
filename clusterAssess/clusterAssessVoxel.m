function [purityIndex1, purityIndex2] = clusterAssessVoxel(xyz, l, resAll, thresh)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% allPurities = clusterAssessKden(xyz, l, kAll)
% 
% Density-based evaluation of clustering:
% We compute the percentage of Pure Voxels: those which have occupancies
% of same label
%
% Input Arguments: 
%
%   xyz    = features, as N x F matrix 
%               
%   l      = labels of points, as N x 1 !!!! positive integer !!! vector
%   resAll = all voxel grid sizes(1 x k vector)
%   thresh = tolerance for unity (scalar, default 1)
%
% Output arguments: 
%
%   purityIndex1 = percentage of (almost) homegeneous cells -> thresh
%   purityIndex2 = Number of datapoints coinciding with index of THEIR cell
%
% Problem about this function:
%
% 1) For many clustering algorithms, density is an intrinsic conditions. 
% That is, many times, you will get excellent results. However, since
% cluster labels do not have anything to do with real classes, the
% classification is then quite bad.
% 
% Then you must use mutual information / adjusted rand score and so on
%
% 2) Only 2 or 3 dimentions are possible to use
%
% Otherwise: Use clusterAssessKdens
% 
% Keywords: @_cluster, @_assess, @_histogram @_cell
%
% Dimitri Bulatov, March 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input arguments
if nargin == 3
    thresh = 1;
end

count = 0;
purityIndex1 = zeros(1, numel(resAll));
purityIndex2 = zeros(1, numel(resAll));
%% Main loop 
for res = resAll
    count = count + 1;
    if size(xyz, 2) == 3
        allCoordIdx = allocatePointsInCubes_v2(xyz', res);
    elseif size(xyz, 2) == 2
        [~, allCoordIdx] = allocatePointsInCells_v2([xyz zeros(size(xyz, 1), 1)]', res);
    else
        error('The number of features must be 2 or 3');
    end
    % Points in each cell, suppress empty cells
    numPoints = cellfun(@numel, allCoordIdx);
    allCoordIdx = allCoordIdx(numPoints > 0);
    
    % Labels in each cell, most frequent one
    allLabelIdx = cellfun(@(x) l(x), allCoordIdx, 'UniformOutput', 0);
    [allWinners, allModes] = cellfun(@(x) mode(x), allLabelIdx);
    
    % compute purity as the percentage of ?relatively? pure cells 
    allPurity = allModes./numPoints(numPoints > 0);
    %figure(1); histogram(allPurity, 20); hold on
    purityIndex1(count) = sum(allPurity > thresh)/numel(allPurity);
    
    % compute purity as accuracy measures
    allAccuracy = cellfun(@(x,y) x == y, allLabelIdx, num2cell(allWinners), 'UniformOutput', false);
    purityIndex2(count) = sum(cell2mat(allAccuracy))/size(xyz, 1);
    
    
    %disp(num2str([res purityIndex1 purityIndex2]));
end
