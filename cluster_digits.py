import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox
from sklearn import cluster, datasets, manifold, metrics, decomposition
from sklearn.preprocessing import MinMaxScaler

NCLASS = 10
# Color for each category
category_colors = plt.get_cmap('tab10')(np.linspace(0., 1., NCLASS))
digit_styles = {'weight': 'bold', 'size': 8}

def plot2D(X, labels, images, title="", save=""):
    fig = plt.figure(figsize=(6, 6), dpi=320)
    ax = fig.add_subplot(1, 1, 1)
    X_std = MinMaxScaler().fit_transform(X)
    
    for xy, l in zip(X_std, labels):
        ax.text(*xy, str(l), color=category_colors[l], **digit_styles)

    image_locs = np.ones((1, 2), dtype=float)
    for xy, img in zip(X_std, images):
        dist = np.sqrt(np.sum(np.power(image_locs - xy, 2), axis=1))
        if np.min(dist) < .05:
            continue
        thumbnail = offsetbox.OffsetImage(img, zoom=.8, cmap=plt.cm.gray_r)
        imagebox = offsetbox.AnnotationBbox(thumbnail, xy)
        ax.add_artist(imagebox)
        image_locs = np.vstack([image_locs, xy])
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save)
    
digits = datasets.load_digits(n_class=NCLASS)
X = digits.data
#PCA
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X)
#plot2D(X_pca, digits.target, digits.images, title="PCA",save="./PCA.png")
#t-SNE
tsne = manifold.TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
#plot2D(X_tsne, digits.target, digits.images, title="t-SNE",save="./t-SNE.png")

km=cluster.KMeans(n_clusters=10, random_state=1)
km_pca=km.fit(X_pca)
SC_pca=metrics.silhouette_score(X_pca,km_pca.labels_,metric='euclidean')
km=cluster.KMeans(n_clusters=10, random_state=1)
km_tsne=km.fit(X_tsne)
SC_tsne=metrics.silhouette_score(X_tsne,km_tsne.labels_,metric='euclidean')
km=cluster.KMeans(n_clusters=10, random_state=1)
km_raw=km.fit(X)
SC_raw=metrics.silhouette_score(X,km_raw.labels_,metric='euclidean')

print('the Silhouette Coefficient of K-means after PCA: '+ str(round(SC_pca,3)))
print('the Silhouette Coefficient of K-means after t-SNE: '+ str(round(SC_tsne,3)))
print('the Silhouette Coefficient of K-means of raw data: '+ str(round(SC_raw,3)))
print('the homogeneity_score of K-means after PCA: '+ str(round(metrics.homogeneity_score(digits.target, km_pca.labels_),3)))
print('the homogeneity_score of K-means after t-SNE: '+ str(round(metrics.homogeneity_score(digits.target, km_tsne.labels_),3)))
print('the homogeneity_score of K-means of raw data: '+ str(round(metrics.homogeneity_score(digits.target, km_raw.labels_),3)))
print('the completeness_score of K-means after PCA: '+ str(round(metrics.completeness_score(digits.target, km_pca.labels_),3)))
print('the completeness_score of K-means after t-SNE: '+ str(round(metrics.completeness_score(digits.target, km_tsne.labels_),3)))
print('the completeness_score of K-means of raw data: '+ str(round(metrics.completeness_score(digits.target, km_raw.labels_),3)))
print('the v_measure_score of K-means after PCA: '+ str(round(metrics.v_measure_score(digits.target, km_pca.labels_),3)))
print('the v_measure_score of K-means after t-SNE: '+ str(round(metrics.v_measure_score(digits.target, km_tsne.labels_),3)))
print('the v_measure_score of K-means of raw data: '+ str(round(metrics.v_measure_score(digits.target, km_raw.labels_),3)))

N=[]
D={0:'Homogeneity',1:'Completeness',2:'V_measure'}
Score=[[],[],[]]
for i in range(64):
    N.append(i+1)
    pca = decomposition.PCA(n_components=i+1)
    X_pca = pca.fit_transform(X)
    km=cluster.KMeans(n_clusters=10, random_state=1)
    km_pca=km.fit(X_pca)
    Score[0].append(metrics.homogeneity_score(digits.target, km_pca.labels_))
    Score[1].append(metrics.completeness_score(digits.target, km_pca.labels_))
    Score[2].append(metrics.v_measure_score(digits.target, km_pca.labels_))
    if i in [10,20,30,40,50,60]:
        print(metrics.confusion_matrix(digits.target, km_pca.labels_))
print(Score)

for i in range(3):
    plt.figure()
    plt.plot(N,Score[i],color='#4B0082')
    plt.xlabel('Dimension')
    plt.ylabel(D[i])
    plt.savefig(D[i]+'.png')




    










