# ----------------------------------------------------------------------------------------------------------------- #
# Bibliotecas
# ----------------------------------------------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import SnowballStemmer

import gensim
from gensim.models import Word2Vec

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# baixar stopwords e pacotes do nltk
# punkt sequence tokenizer = divide um texto em uma lista de sentenças usando um algoritmo não supervisionado 
# para construir um modelo para abreviação de palavras

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('rslp')

# ----------------------------------------------------------------------------------------------------------------- #
# Variáveis
# ----------------------------------------------------------------------------------------------------------------- #

stop_words = set(stopwords.words('portuguese'))

# ----------------------------------------------------------------------------------------------------------------- #
# Stemming
# ----------------------------------------------------------------------------------------------------------------- #

stemmer = nltk.stem.RSLPStemmer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

# ----------------------------------------------------------------------------------------------------------------- #
# Quantidade de palavras distintas
# ----------------------------------------------------------------------------------------------------------------- #

def qtd_palavras_distintas(df, coluna):
    # Converte todos os comentários em uma única string
    todos_comentarios = " ".join(df[coluna])

    # Tokeniza os comentários para obter uma lista de palavras
    palavras = word_tokenize(todos_comentarios)

    # Remove palavras duplicadas transformando em um set, e conta o total de palavras distintas
    palavras_distintas = set(palavras)
    quantidade_palavras_distintas = len(palavras_distintas)

    print(f"Quantidade de palavras distintas: {quantidade_palavras_distintas}")

# ----------------------------------------------------------------------------------------------------------------- #
# Obter vetor word2vec
# ----------------------------------------------------------------------------------------------------------------- #

def get_average_word2vec(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis = 0) if vectors else np.zeros(100)

# ----------------------------------------------------------------------------------------------------------------- #
# Gráfico elbow method
# ----------------------------------------------------------------------------------------------------------------- #

def plot_elbow_method(X, max_k = 10, title = "Elbow Method"):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters = k, random_state = 42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize = (16, 5))
    plt.plot(range(1, max_k + 1), inertias, marker = 'o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title(title)

    # Configura o range do eixo y
    plt.ylim(min(inertias) - 1000, max(inertias) + 1000)

    # Configura o eixo x para ir de 1 em 1
    plt.xticks(range(1, max_k + 1, 1))

    plt.show()

# ----------------------------------------------------------------------------------------------------------------- #
# Gráfico Elbow Method
# ----------------------------------------------------------------------------------------------------------------- #

def plot_elbow_method(X, max_k = 10, title = "Elbow Method"):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters = k, random_state = 42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize = (16, 5))
    plt.plot(range(1, max_k + 1), inertias, marker = 'o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title(title)

    # Configura o range do eixo y
    plt.ylim(min(inertias) - 1000, max(inertias) + 1000)

    # Configura o eixo x para ir de 1 em 1
    plt.xticks(range(1, max_k + 1, 1))

    plt.show()

# ----------------------------------------------------------------------------------------------------------------- #
# Gráfico Silhouette Scores
# ----------------------------------------------------------------------------------------------------------------- #

def plot_silhouette_scores(X, max_k = 10, title = "Elbow Method"):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters = k, random_state = 42)
        kmeans.fit_predict(X)
        # print(silhouette_score(X, kmeans.labels_))
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    plt.figure(figsize = (16, 5))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker = 'o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title(title)

    # Configura o range do eixo y
    plt.ylim(min(silhouette_scores) - 0.1, max(silhouette_scores) + 0.1)

    # Configura o eixo x para ir de 1 em 1
    plt.xticks(range(1, max_k + 1, 1))

    # Adiciona os rótulos de dados para cada ponto com 3 casas decimais
    for i, score in enumerate(silhouette_scores, start=2):
        plt.text(i, score, f"{score:.3f}", ha='center', va='bottom', fontsize=10, color='black')

    plt.show()

# ----------------------------------------------------------------------------------------------------------------- #
# Palavra mais comum de cada cluster
# ----------------------------------------------------------------------------------------------------------------- #

def nomes_clusters(df, coluna):

    most_common_words = {}

    for cluster in df[coluna].unique():
        texts_in_cluster = df[df[coluna] == cluster]['preprocessed_text']
        words = " ".join(texts_in_cluster).split()
        word_counts = Counter(words)
        most_common_word = word_counts.most_common(30)#[0]
        most_common_words[cluster] = most_common_word

    tmp = df[coluna].value_counts().reset_index()
    tmp['top_palavras'] = tmp[coluna].map(most_common_words).apply(lambda lista: [t[0] for t in lista])
    
    # palavras exclusivas
    todas_palavras = set(palavra for lista in tmp['top_palavras'] for palavra in lista)

    def palavras_unicas(lista_palavras, todas_palavras, listas_restantes):
        palavras_outras_linhas = set(palavra for lista in listas_restantes for palavra in lista)
        return [palavra for palavra in lista_palavras if palavra not in palavras_outras_linhas]

    tmp['palavras_exclusivas_top'] = tmp.apply(
        lambda row: palavras_unicas(row['top_palavras'], todas_palavras, tmp[tmp.index != row.name]['top_palavras']),
        axis=1
    )

    tmp = tmp[[coluna, 'top_palavras', 'palavras_exclusivas_top', 'count']]

    return tmp


# ----------------------------------------------------------------------------------------------------------------- #
# Visualização dos clusters (PCA)
# ----------------------------------------------------------------------------------------------------------------- #

def pca_plot(metodo):

    print(f"Silhouette Score {metodo}: {round(silhouette_score(eval(f'X_{metodo}'), eval(f'kmeans_{metodo}').labels_), 5)}")

    # Número de clusters
    num_clusters = eval(f'n_clusters_{metodo}')

    # Reduzindo a dimensionalidade para 2D com PCA
    pca = PCA(n_components = 2, random_state = 42)
    
    if type(eval(f'X_{metodo}')) == np.ndarray:
        X_array = eval(f'X_{metodo}')
    else:
        X_array = eval(f'X_{metodo}').toarray()
    
    X_2d = pca.fit_transform(X_array)

    # Aplicando o KMeans e pegando as previsões
    kmeans_pca = eval(f'kmeans_{metodo}')
    clusters = kmeans_pca.fit_predict(eval(f'X_{metodo}'))

    # Plotando o scatter plot
    plt.figure(figsize = (10, 7))
    for cluster in range(num_clusters):
        # Filtrando os pontos de cada cluster
        plt.scatter(X_2d[clusters == cluster, 0], X_2d[clusters == cluster, 1],
                    label=f"Cluster {cluster}", alpha=0.6)

    # Plotando os centróides
    centroids_2d = pca.transform(kmeans_pca.cluster_centers_)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s = 100, c = 'black', marker = 'X', label = "Centroids")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.title(f"Clusters ({metodo.capitalize()})")
    plt.show()

# ----------------------------------------------------------------------------------------------------------------- #
# Gráfico Silhouette Coeficient
# ----------------------------------------------------------------------------------------------------------------- #

# def silhouette_plot(metodo, n_clusters = None):
#     X = eval(f'X_{metodo}')

#     if n_clusters == None:
#         n_clusters = eval(f'n_clusters_{metodo}')

#     pca = PCA(n_components=2)

#     if type(eval(f'X_{metodo}')) == np.ndarray:
#         X_array = eval(f'X_{metodo}')
#     else:
#         X_array = eval(f'X_{metodo}').toarray()

#     X_reduced = pca.fit_transform(X_array)

#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(20, 8)

#     # O primeiro subplot é o gráfico de Silhouette
#     ax1.set_xlim([-0.1, 1])
#     ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])

#     # Inicializa o KMeans com n_clusters
#     kmeans_silhouette = eval(f'kmeans_{metodo}')
#     cluster_labels = kmeans_silhouette.fit_predict(eval(f'X_{metodo}'))

#     # Calcula o Silhouette Score médio para todos os pontos
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print(f"Para n_clusters = {n_clusters}, o Silhouette Score médio é: {round(silhouette_avg, 5)}")

#     # Calcula os scores de silhouette para cada amostra
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Obtém os valores de silhouette para amostras pertencentes ao cluster i, e ordena
#         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(
#             np.arange(y_lower, y_upper),
#             0,
#             ith_cluster_silhouette_values,
#             facecolor=color,
#             edgecolor=color,
#             alpha=0.7,
#         )

#         # Rotula os gráficos de silhouette com o número do cluster
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Calcula y_lower para o próximo gráfico
#         y_lower = y_upper + 10  # 10 para o espaço entre clusters

#     ax1.set_title("Silhouette plot for the various clusters.")
#     ax1.set_xlabel("Silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")

#     # Linha vertical para o Silhouette Score médio de todos os valores
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#     ax1.axvline(x=0, color="black", linestyle="--")
#     ax1.set_yticks([])
#     ax1.set_xticks([# -1, -0.8, -0.6, -0.4, 
#                     -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

#     # Segundo plot mostrando os clusters formados
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(
#         X_reduced[:, 0], X_reduced[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
#     )

#     # Marcando os centróides dos clusters
#     centers = pca.transform(kmeans_silhouette.cluster_centers_)
#     ax2.scatter(
#         centers[:, 0],
#         centers[:, 1],
#         marker="o",
#         c="white",
#         alpha=1,
#         s=200,
#         edgecolor="k",
#     )

#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")

#     ax2.set_title("Visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")

#     plt.suptitle(
#         f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
#         fontsize=14,
#         fontweight="bold",
#     )

# plt.show()