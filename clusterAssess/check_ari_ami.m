pythonPath = 'c:\Daten\Python37\Data\';


core = rand(100, 200);
mask = ceil(core*10);
indicesV1 = max(1, ceil(core*12+randn(100, 200)*0.1));
%save ([pythonPath 'synthetic'], 'mask', 'indicesV1');
%load ([pythonPath 'clusterAssess_lab'], 'indicesV1', 'indicesV2', 'indices1_MRF_mp', 'indices1_MRF_mm', 'mask', 'farbwerte', 'zeilen', 'spalten');

% [AMI_]=ami(mask(:),indicesV1(:)); % perfect coincidence


ami =ami(mask(:),indicesV1(:));

ari = rand_index(mask(:), indicesV1(:), 'adjusted');



save ([pythonPath 'synthetic'], 'mask', 'indicesV1');