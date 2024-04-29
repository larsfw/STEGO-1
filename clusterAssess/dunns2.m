function d = dunns2(feat, cluster_number, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 
%
% Dunns function for cluster assessing
%
% d = max(dist(intraclass))/min(dist(interclass))
%
% In comparison to the original implementation, it has been extensively modified. 
% 	The highest intraclass distance was obtained using convex hulls.
% 	The lowest interclass distance was obtained using KNN algorithm for two point clouds
%
% Input: 
%
% 	feat 		= featureSet (N x D, whereby N is the number of datapoints)
%	cluster_number 	= label of the cluster (scalar)
%	varargin 	= typical for distance measurement list of arguments
%
% Output: 
%	d 		= Dunn's measure
%
% Minimum example: dunnsDemo.m
%
% To do: Interesting topic of hierarchcal computation of dist(interclass) for each cluster
%	 
% Keywords: @_clustering, @_validation, @_ML	 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input
if nargin == 3
    varargin2{1} = 'Distance';
    varargin2 = [varargin2, varargin];
else
    varargin2 = varargin;
end
% pre-process: sorting
[~, xi] = sort(cluster_number); 
feat = feat(xi, :);
cluster_number = cluster_number(xi);

% determine the number of clusters
idx = [0; find(diff(cluster_number) > 0); numel(cluster_number)];
numCluster = numel(idx) - 1;
% allocate
num = zeros(1, numCluster); dem = num;
for ii = 1:numCluster
    % points in this cluster:
    X = feat(idx(ii)+1:idx(ii+1), :);
    % points in other clusters
    Y = feat([0+1:idx(ii) idx(ii+1)+1:end], :);
    
    % intra-cluster distance
    [~, dIntra] = knnsearch(Y, X, 'K', 1, varargin2{:});
    num(ii) = min(dIntra);
    
    % inter-cluster distance
    q = convhulln(X);
    Xq = X(unique(q(:)), :);
    dInter = pdist(Xq, varargin{:});
    dem(ii) = max(dInter(:));
end

d = min(num)/max(dem);
    
    

