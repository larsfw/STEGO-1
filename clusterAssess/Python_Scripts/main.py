# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import matplotlib.path as mplt_path
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import io, stats
import numpy as np
from skimage import segmentation, color
from sklearn import ensemble, metrics
from display_project.GinLab import GinLab
#import cv2


print("Hallo")
# a) Laden der Daten:
myPath = 'd:/buldim/Data/Python/TestData/'
V = io.loadmat(myPath + 'clusterAssess.mat')




# variablen:
mask = V['mask']
mask = mask.reshape(1, mask.size)
indices = V['indicesV1'].reshape(1, mask.size)
numL = np.max(indices)
colorValues = V['farbwerte']


GinLab.run([V['mask'], V['indicesV1']])

A = metrics.adjusted_rand_score(mask[0, :], indices[0, :])
B = metrics.adjusted_mutual_info_score(mask[0, :], indices[0, :])
print('adjusted rand / mi score for k-means: ' + str(np.ceil(1000*A)/1000) + ' / ' + str(np.ceil(1000*B)/1000))

D = metrics.davies_bouldin_score(colorValues,indices[0, :])
C = metrics.silhouette_score(colorValues,indices[0,:])


print('davies-bouldin / silhouette score at random points: ' + str(np.ceil(1000*D)/1000) + ' / ' + str(np.ceil(1000*C)/1000))


indices = V['indicesV2'].reshape(1, mask.size)
A = metrics.adjusted_rand_score(mask[0, :], indices[0, :])
B = metrics.adjusted_mutual_info_score(mask[0, :], indices[0, :])

print('adjusted rand / mi score for k-means with neighborhoods: ' + str(np.ceil(1000*A)/1000) + ' / ' + str(np.ceil(1000*B)/1000))

D = metrics.davies_bouldin_score(colorValues,indices[0, :])
C = metrics.silhouette_score(colorValues,indices[0,:])
print('davies-bouldin / silhouette score at random points: ' + str(np.ceil(1000*D)/1000) + ' / ' + str(np.ceil(1000*C)/1000))


indices = V['indices1_MRF_mm'].reshape(1, mask.size)
A = metrics.adjusted_rand_score(mask[0, :], indices[0, :])
B = metrics.adjusted_mutual_info_score(mask[0, :], indices[0, :])
print('adjusted rand / mi score for k-means with MRF Move Making: ' + str(np.ceil(1000*A)/1000) + ' / ' + str(np.ceil(1000*B)/1000))

D = metrics.davies_bouldin_score(colorValues,indices[0, :])
C = metrics.silhouette_score(colorValues,indices[0,:])
print('davies-bouldin / silhouette score at random points: ' + str(np.ceil(1000*D)/1000) + ' / ' + str(np.ceil(1000*C)/1000))


indices = V['indices1_MRF_mp'].reshape(1, mask.size)
A = metrics.adjusted_rand_score(mask[0, :], indices[0, :])
B = metrics.adjusted_mutual_info_score(mask[0, :], indices[0, :])
print('adjusted rand / mi score for k-means with MRF Message Passing: ' + str(np.ceil(1000*A)/1000) + ' / ' + str(np.ceil(1000*B)/1000))

D = metrics.davies_bouldin_score(colorValues,indices[0, :])
C = metrics.silhouette_score(colorValues,indices[0,:])
print('davies-bouldin / silhouette score at random points: ' + str(np.ceil(1000*D)/1000) + ' / ' + str(np.ceil(1000*C)/1000))



#%% Let's consider extremal cases:
indices[0,:] = np.round(np.random.rand(indices.size) * (numL - 1)).astype(int)
A = metrics.adjusted_rand_score(mask[0, :], indices[0, :])
B = metrics.adjusted_mutual_info_score(mask[0, :], indices[0, :])
print('adjusted rand / mi score for random image: ' + str(np.ceil(1000*A)/1000) + ' / ' + str(np.ceil(1000*B)/1000))

D = metrics.davies_bouldin_score(colorValues,indices[0, :])
C = metrics.silhouette_score(colorValues,indices[0,:])
print('davies-bouldin / silhouette score at random points: ' + str(np.ceil(1000*D)/1000) + ' / ' + str(np.ceil(1000*C)/1000))


indices = mask
A = metrics.adjusted_rand_score(mask[0, :], indices[0, :])
B = metrics.adjusted_mutual_info_score(mask[0, :], indices[0, :])
print('adjusted rand / mi score for ground truth: ' + str(np.ceil(1000*A)/1000) + ' / ' + str(np.ceil(1000*B)/1000))

D = metrics.davies_bouldin_score(colorValues,indices[0, :])
C = metrics.silhouette_score(colorValues,indices[0,:])
print('davies-bouldin / silhouette score at random points: ' + str(np.ceil(1000*D)/1000) + ' / ' + str(np.ceil(1000*C)/1000))

