"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

############## Task 1

## Please change directory accordingly
os.chdir("/Users/celinehajjar/Desktop/MVA/altegrad/Lab6_hajjar_celine/code/part1")

G = nx.read_edgelist("../datasets/CA-HepTh.txt",delimiter="\t")

print("Number of edges: ", G.number_of_edges())
print("Number of nodes: ", G.number_of_nodes())


############## Task 2

# your code here #

print("Number of connected components: ", nx.number_connected_components(G))

#Largest connected component
largest_cc = max(nx.connected_components(G), key=len)
largest_cc_subgraph =  G.subgraph(largest_cc).copy()

print("Number of edges in the largest connected component: ",largest_cc_subgraph.number_of_edges())

print("Number of nodes in the largest connected component: ",largest_cc_subgraph.number_of_nodes())

ratio_edges = largest_cc_subgraph.number_of_edges()/G.number_of_edges()
ratio_nodes = largest_cc_subgraph.number_of_nodes()/G.number_of_nodes()
print("Ratio of edges w.r.t to the whole graph:{:.2f}".format(ratio_edges))
print("Ratio of nodes w.r.t to the whole graph:{:.2f}".format(ratio_nodes))

#We see that even though there are a lot of components, most of the edges and nodes are contained in only one of them: the others must be small

############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

# your code here #
print("minimum node degree: ", np.min(degree_sequence))
print("maximum node degree: ", np.max(degree_sequence))
print("median node degree: ", np.median(degree_sequence))
print("mean node degree: {:.2f}".format(np.mean(degree_sequence)))



############## Task 4

# your code here #


plt.figure(figsize=(15, 6))
plt.plot(nx.degree_histogram(G))
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.title("Distribution of Node Degrees")
plt.show()

plt.figure(figsize=(15, 6))
plt.yscale('log')
plt.xscale('log')
plt.plot(nx.degree_histogram(G))
plt.ylabel('log(frequency)')
plt.xlabel('log(degree)')
plt.title("Logarithmic Distribution of Node Degrees")
plt.show()

############## Task 5


# your code here #


print("The global clustering coefficient of the graph is:{:.2f} ".format(nx.transitivity(G)))