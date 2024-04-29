function [allPurities, allIsPure] = clusterAssessKdens(xyz, l, kAll)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% allPurities = clusterAssessKdens(xyz, l, kAll)
%
% Density-based evaluation of clustering:
% We compute the percentage of Pure points: those which have their
% nearest neighbors of same label
%
% Input Arguments:
%
%   xyz   = features, as N x F matrix
%           
%   l     = labels of points, as N x 1 !!!! positive integer !!! vector
%   
%   kAll  = all numbers of nearest neighbors (1 x k vector)
%
% Output arguments:
%
%   allPurities = for every k, we get purities
%
%   allIsPure   = binary matrix, which points have only neighbors from the
%                 same class
%
% Problems about this function:
% 1) For many clustering algorithms, density is an intrinsic conditions.
% That is, many times, you will get excellent results. However, since
% cluster labels do not have anything to do with real classes, the
% classification is then quite bad.
%
% 2) If the sensitivity on k is too strong -> please use clusterAssessVoxel
%
% To do: maybe introduce a thresh
%        since idx does not depend on k, maybe pre-compute?
%
% @_cluster, @_assessment, @_knn
%
% Dimitri Bulatov, March 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Preparation:
k = max(kAll);
% if isstruct(xyz) && isfield(xyz, 'idx') && size(xyz.idx, 2) >= k && size(xyz, 1) == numel(l)
%     idx = xyz.idx;
% else
% we have the possibility to compute neighbors for quite large value of k
[idx, dst] = knnsearch(xyz, xyz, 'K', k);
%end
labelIdx = l(idx);
% allocate space for the output
allPurities = zeros(1, numel(kAll));
allIsPure = zeros(size(xyz, 1), numel(kAll));

count = 0;

%% Main loop
for kk = kAll
    count = count+1;
    allMin = min(labelIdx(:, 1:kk), [], 2);
    allMax = max(labelIdx(:, 1:kk), [], 2);
    isPure = allMin == allMax;
    allPurities(count) = sum(isPure)/size(xyz, 1);
    allIsPure(:, count) = isPure;
end
