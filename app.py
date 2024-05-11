import streamlit as st
import os
from extract_and_cluster import (extract_texts_from_pdfs, find_top_similar_papers, get_embeddings, cluster_embeddings_dbscan, cluster_embeddings_kmeans, 
                                cluster_embeddings_spectral, cluster_embeddings_gmm, cluster_embeddings_agglomerative,
                                cluster_embeddings_optics)
import fitz
import os
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from random import sample


def visualize_clusters(filenames, labels):
    G = nx.Graph()
    for filename, label in zip(filenames, labels):
        G.add_node(filename, group=label)

    # Add edges within the same cluster
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                G.add_edge(filenames[i], filenames[j])

    pos = nx.spring_layout(G)
    colors = [G.nodes[node]['group'] for node in G.nodes()]
    ff = plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_color=colors, with_labels=True, font_size=5, cmap=plt.cm.Set1)
    st.pyplot(ff)  # Use Streamlit's function to display the plot

def find_top_similar_papers_and_plot(embeddings, filenames):
    # calculate the pairwise cosine similarities
    sim_matrix = cosine_similarity(embeddings)

    # select two random papers
    selected_indices = sample(range(len(filenames)), 2)
    fig, axes = plt.subplots(1, 2, figsize=(40, 20)) # play with subplot size to make stuff fit

    # process each selected paper
    for idx, paper_index in enumerate(selected_indices):
        # get the indices of the top 6 scores (including itself)
        similar_indices = np.argsort(-sim_matrix[paper_index])[:6]
        
        # remove the paper itself from the list
        top_indices = [i for i in similar_indices if i != paper_index][:5]

        # create graph
        G = nx.Graph()
        # add edges between the selected paper and the top 5 similar papers
        for rank, i in enumerate(top_indices):
            G.add_edge(filenames[paper_index], filenames[i], weight=sim_matrix[paper_index][i])

        # position nodes with the spring layout
        pos = nx.spring_layout(G)
        
        # draw
        selected_title = filenames[paper_index][:15]  # shorten
        labels = {filenames[paper_index]: f'Sel: {selected_title}'}
        for i, node in enumerate(top_indices, start=1):
            short_title = filenames[node][:15] 
            labels[filenames[node]] = f'({i}) {short_title}'

        nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=3000, font_size=9, ax=axes[idx])
        axes[idx].set_title(f'Top 5 Similar Papers to "{selected_title}"')

    st.pyplot(fig)  # Use Streamlit's function to display the plot


def load_data():
    directory = 'papers'
    if not os.path.exists(directory):
        os.makedirs(directory)
        st.sidebar.write(f'Please upload PDF files to the "{directory}" directory.')
        return None, None
    texts, filenames = extract_texts_from_pdfs(directory)
    embeddings = get_embeddings(texts)
    return embeddings, filenames

def main():
    st.title("Research Paper Analysis App")
    
    embeddings, filenames = load_data()

    if embeddings is None or filenames is None:
        st.sidebar.write("Waiting for data...")
        return

    clustering_option = st.sidebar.selectbox(
        'Choose a clustering method:',
        ('DBSCAN', 'K-Means', 'Spectral Clustering', 'Gaussian Mixture', 'Agglomerative Clustering', 'OPTICS')
    )

    if st.sidebar.button('Cluster'):
        if clustering_option == 'DBSCAN':
            labels = cluster_embeddings_dbscan(embeddings)
        elif clustering_option == 'K-Means':
            labels = cluster_embeddings_kmeans(embeddings)
        elif clustering_option == 'Spectral Clustering':
            labels = cluster_embeddings_spectral(embeddings)
        elif clustering_option == 'Gaussian Mixture':
            labels = cluster_embeddings_gmm(embeddings)
        elif clustering_option == 'Agglomerative Clustering':
            labels = cluster_embeddings_agglomerative(embeddings)
        elif clustering_option == 'OPTICS':
            labels = cluster_embeddings_optics(embeddings)
        
        visualize_clusters(filenames, labels)

    if st.sidebar.button('Find Similar Papers'):
        results = find_top_similar_papers_and_plot(embeddings, filenames)
        # st.write(results)

if __name__ == '__main__':
    main()
