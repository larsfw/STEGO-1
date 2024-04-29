function [dunnsIndex, purityIndex1, purityIndex2, densIndex, arIndex, amiIndex] = clusterAssessAllMeasures(farbwerte, indices1, mask, allV, allK, threshV)
%%%%%%%%%%%%%%%%%%%%%%%%
%
% [dunnsIndex, purityIndex1, purityIndex2, densIndex, myari, myami] = clusterAssessAllMeasures(farbwerte, indices1, mask, allV, allK, threshV)
%
% Compilation of important clustering measures
%
% Input arguments: 
%
%     farbwerte: = Feature, N x 3
%     indices1   = Clustering Indices N x 1
%     mask       = True indices N x 1
%     allV       = Voxel grid size (s), 
%                  1 x L beware memory explosion or empty 
%     allK       = k-Indices, 1 x L or empty (in this case, 0 is output)
%     threshV    = Threshold for pure voxels
% 
% Without externally labeled reference:
%
%   Dunns Metric [To demonstrate Dimis acceleration]
%   Voxel based Metric
%   Density based metric
%
% NOT
%
%   Silhouette [Needs too long]
%
% With externally labeled reference:
%
%   Adjusted rand index (ari)
%   Adjusted mutual information (ami)
%
% Dimitri Bulatov, July 22
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Unsupervised metrix
% dunn: faster way
dunnsIndex   = dunns2(farbwerte, indices1(:));

% Voxel-based
if ~isempty(allV) 
    if(size(farbwerte,2) == 2 || size(farbwerte,2) == 3) 
        [purityIndex1, purityIndex2] = clusterAssessVoxel(farbwerte, indices1, allV, threshV);
    else
        purityIndex1 = zeros(1, numel(allV));
        purityIndex2 = zeros(1, numel(allV));
    end
else
    purityIndex1 = 0;
    purityIndex2 = 0;
end

% K-based (density)
if ~isempty(allK)
    densIndex = clusterAssessKdens(farbwerte, indices1, allK);
else
    densIndex = 0;
end
%silhIndex = median(silhouette(farbwerte, indices1, 'Euclidean'));
%% Hier wird ein WS an Denis abgeschickt, damit er ARS und AMI rechnet?
% Does NOT depend on features!
if ~isempty(mask)
    amiIndex =ami(mask(:),indices1(:));
    arIndex = rand_index(mask(:), indices1(:), 'adjusted');
elseif nargout > 4
    error('Mask as array of the same size as Clustering result is needed');
end