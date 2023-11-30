from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import prince
from sklearn.cluster import KMeans
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def dim_red_umap(mat, p):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    
    # Create UMAP model and fit_transform the embeddings to reduce to 20 dimensions
    umap_model = UMAP(n_components=p)
    umap_result = umap_model.fit_transform(embeddings)

    # red_mat = mat[:,:p]
    
    return umap_result


def dim_red_Acp(mat, p):
    '''
    Perform dimensionality reduction

    Input:
    -----

        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    
    red_mat = mat[:,:]
    red_mat = pd.DataFrame(red_mat)
    pca = prince.PCA(n_components=p)
    pca = pca.fit(red_mat)
    
    return pca.transform(red_mat)

def dim_red_TSNE(mat,p) :
    '''
    mat : NxM list 
    p : number of dimensions to keep 
    Output:
    ------
    red_mat : NxP list such that p<<m
    '''
    # Normaliser les données
    scaler = StandardScaler()
    data_std = scaler.fit_transform(mat)

    # Instancier t-SNE
    tsne_model = TSNE(n_components=p, random_state=42)

    # Ajuste le modèle 
    resultat_tsne = tsne_model.fit_transform(data_std)

    tsne_df = pd.DataFrame(resultat_tsne, columns=[f"Dimension_{i}" for i in range(1, p + 1)])
    red_mat = tsne_df

    return red_mat


def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        red_mat = dim_red_Acp(mat, p)
        
    elif method=='TSNE':
        red_mat = dim_red_TSNE(mat,p)
       
    elif method=='UMAP':
        red_mat = dim_red_umap(mat,p)
        
    else:
        raise Exception("Please select one of the three methods : APC, TSNE, UMAP")
    
    return red_mat


def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    # Instancier Kmeans
    model_kmeans = KMeans(n_clusters=k, random_state=42)

    # Ajuster le modèle 
    model_kmeans.fit(mat)

    pred = model_kmeans.labels_


    return pred

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'TSNE', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')