"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans

#### Please change directory accordingly

os.chdir("/Users/celinehajjar/Desktop/MVA/altegrad/Lab6_hajjar_celine/code/part1")


G = nx.read_edgelist("../datasets/CA-HepTh.txt",delimiter="\t")
largest_cc = max(nx.connected_components(G), key=len)
largest_cc_subgraph =  G.subgraph(largest_cc).copy()

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
    
    ##################
    # your code here #
    ##################
    
def spectral_clustering(G, k):
    A = nx.adjacency_matrix(G)
    D_inv = diags([1/G.degree(node) for node in G.nodes()])
    L = eye(G.number_of_nodes())- D_inv@A
     
    eigval,eigvec = eigs(L, which='SR', k=k)
    eigvec = np.real(eigvec)
    
    kmeans = KMeans(k).fit(eigvec)
    clustering={}
    for i,node in enumerate(G.nodes()):
        clustering[node]=kmeans.labels_[i]
        
    return clustering



############## Task 7

##################
# your code here #
##################

clustering_largest = spectral_clustering(largest_cc_subgraph, 50)

############## Task 8
# Compute modularity value from graph G based on clustering
    
    ##################
    # your code here #
    ##################
def modularity(G, clustering):
    clusters = set(clustering.values())
    m=G.number_of_edges()
    modularity=0
    for cluster in clusters:
        nodes_in_cluster = [node for node in G.nodes() if clustering[node]==cluster]
        cluster_graph = G.subgraph(nodes_in_cluster)
        lc = cluster_graph.number_of_edges()
        dc = 0
    
        for node in nodes_in_cluster:
            dc+=G.degree(node)
        modularity += lc/m - (dc/(2*m))**2
        
    return modularity



############## Task 9

##################
# your code here #
##################

print(modularity(largest_cc_subgraph, clustering_largest))

random_clustering = {}
for node in G.nodes():
    random_clustering[node]=randint(0,49)
print(modularity(largest_cc_subgraph, random_clustering))

#Spectral clustering beats the baseline of random clustering