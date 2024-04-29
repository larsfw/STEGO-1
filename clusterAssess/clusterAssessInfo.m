%%%%%%%%%%%%%%%%%%%
% Here we have all the funcitons for assessing the clusters. 
%	(Practically there are two types: 
% 			extrinsic, that is using the extermnal references (like AMI)
%			intrinsic, like Dunns, Silhouette, K-Density, V-Density
%
% They were downloaded from different sources in the internet.
% Partly, they were modified.
%
% So, the AMI function returns MI value as well
%
% Dunns function has been extensively modified. 
% 	The highest intraclass distance was obtained using convex hulls.
% 	The lowest interclass distance was obtained using KNN algorithm for two point clouds
%
%  Keywords: @_cluster, @_valudation, @_maths, @_neighbors
%
% Dimitri Bulatov, November 2022
%
%%%%%%%%%%%%%%%%%%%