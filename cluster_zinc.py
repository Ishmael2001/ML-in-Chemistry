import numpy as np
import pandas as pd
from matplotlib import offsetbox
from sklearn import cluster, datasets, manifold, metrics, decomposition
from sklearn.preprocessing import MinMaxScaler

def tanimoto_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, ord=2) ** 2 + np.linalg.norm(y,ord=2) ** 2 - np.dot(x, y))
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2))
def dice_similarity(x, y):
    return 2 * np.dot(x, y) / (np.linalg.norm(x, ord=1) + np.linalg.norm(y,ord=1))

num=12

smiles = pd.read_csv('../data/zinc_SMILES.csv')
smiles = np.array(smiles)[:,1]
fp = pd.read_csv('../data/zinc_fp.csv')
fp = np.array(fp)[:,1:]

pca = decomposition.PCA(n_components=200)
fp_pca = pca.fit_transform(fp)

#euclidean
tsne1 = manifold.TSNE(n_components=2,metric='euclidean')
fp_tsne1 = tsne1.fit_transform(fp_pca)
km=cluster.KMeans(n_clusters=num, random_state=1)
km_fp1=km.fit(fp_tsne1)
SC_euclidean=metrics.silhouette_score(fp_tsne1,km_fp1.labels_,metric='euclidean')
print('the Silhouette Coefficient using euclidean of K-means: '+str(round(SC_euclidean,3)))
dic={}
i=0
while len(dic)<num:
   if km_fp1.labels_[i] not in dic.keys():
       dic[km_fp1.labels_[i]]=smiles[i]
   i+=1
for i in range(num):
    print('typical SMILES for cluster '+str(i)+' : '+dic[i])

#cosine
tsne2 = manifold.TSNE(n_components=2,metric='cosine')
fp_tsne2 = tsne2.fit_transform(fp_pca)
km=cluster.KMeans(n_clusters=num, random_state=1)
km_fp2=km.fit(fp_tsne2)
SC_cosine=metrics.silhouette_score(fp_tsne2,km_fp2.labels_,metric='euclidean')
print('the Silhouette Coefficient using cosine of K-means: '+str(round(SC_cosine,3)))
dic={}
i=0
while len(dic)<num:
   if km_fp2.labels_[i] not in dic.keys():
       dic[km_fp2.labels_[i]]=smiles[i]
   i+=1
for i in range(num):
    print('typical SMILES for cluster '+str(i)+' : '+dic[i])
    
#dice
tsne3 = manifold.TSNE(n_components=2,metric=dice_similarity)
fp_tsne3 = tsne3.fit_transform(fp_pca)
km=cluster.KMeans(n_clusters=num, random_state=1)
km_fp3=km.fit(fp_tsne3)
SC_dice=metrics.silhouette_score(fp_tsne3,km_fp3.labels_,metric='euclidean')
print('the Silhouette Coefficient using dice of K-means: '+str(round(SC_dice,3)))
dic={}
i=0
while len(dic)<num:
   if km_fp3.labels_[i] not in dic.keys():
       dic[km_fp3.labels_[i]]=smiles[i]
   i+=1
for i in range(num):
    print('typical SMILES for cluster '+str(i)+' : '+dic[i])

#tanimoto
tsne4 = manifold.TSNE(n_components=2,metric=tanimoto_similarity)
fp_tsne4 = tsne4.fit_transform(fp_pca)
km=cluster.KMeans(n_clusters=num, random_state=1)
km_fp4=km.fit(fp_tsne4)
SC_tanimoto=metrics.silhouette_score(fp_tsne4,km_fp4.labels_,metric='euclidean')
print('the Silhouette Coefficient using tanimoto of K-means: '+str(round(SC_tanimoto,3)))
dic={}
i=0
while len(dic)<num:
   if km_fp4.labels_[i] not in dic.keys():
       dic[km_fp4.labels_[i]]=smiles[i]
   i+=1
for i in range(num):
    print('typical SMILES for cluster '+str(i)+' : '+dic[i])

