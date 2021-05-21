"""
    Script encarregue de criar um modelo de aprendizagem não supervisionada
    para agrupar as raças por grupos.

    Ao sabermos quais as raças mais idênticas, consequentemente sabemos as mais distintas

    Posteriormente, podemos verificar se as raças escolhidas pelo algoritmo de similaridade
    por cosseno escolhe realmente raças de grupos distintos
"""
from sklearn.cluster import AgglomerativeClustering # Criar os clusters
from scipy.cluster.hierarchy import dendrogram # Plotar dendogram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%matplotlib inline


def plot_dendogram(cluster, **kwargs):
    """
        Plotar o gráfico com o dendograma
        https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """

    # Número de nós
    n_nodes = np.zeros(cluster.children_.shape[0])
    n_samples = len(cluster.labels_)

    for i, merge in enumerate(cluster.children_):
        cur_count = 0

        for child_idx in merge:
            if child_idx < n_samples:
                cur_count += 1 # Folha
            else:
                cur_count += n_nodes[child_idx - n_samples]

        n_nodes[i] = cur_count
    
    link_matrix = np.column_stack([cluster.children_, cluster.distances_, n_nodes]).astype(float)

    dendrogram(link_matrix, **kwargs)


def append_labels(df, labels):
    """
        Função que recebe um DataFrame e a classe/grupo de cada raça,
        criando e juntando o grupo no Dataframe

        return: DataFrame
            DataFrame com os grupos concatenados
    """

    df["Grupo"] = labels

    return df


def get_clusters(df, cat, n_clusters, prune=3):
    """
        Função feita para treinar um algoritmo de agrupamento (clustering)
        com aprendizagem não supervisionada.
        Poderemos ver o Dendograma para decidir quantos clusters desejamos

        df: DataFrame
            DataFrame com one hot encoding para ser usado no algoritmo
        
        cat: DataFrame
            DataFrame original com todas as features categóricas 

        n_clusters: int
            Número de clusters desejados
        
        prune: int
            Cortar dendograma num certo nível (prune)

        return: DataFrame
            Retorna uma DataFrame vazio caso só queiramos ver o dendograma
            ou então um DataFrame com o grupo(cluster) de cada amostra
    """

    # Remover coluna Raca
    df.drop(["Raca"], axis=1, inplace=True)

    # Converter para um array
    X = df.to_numpy()

    # Objeto do Cluster
    # Vou usar o linkage "complete" para maximizar a distância
    # de cada set

    # Criar loop para saber se queremos plotar o dendograma
    res = ""
    while res != 'n' and res != 'y':
        res = input("Plotar Dendograma [y/n]: ")

        if res == 'y':
            cluster = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="complete")
            cluster.fit(X)

            plot_dendogram(cluster, truncate_mode="level", p=prune)

            df = pd.DataFrame([])

            plt.show()
        
        elif res == 'n':
            cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
            cluster.fit(X)

            df = append_labels(cat, cluster.labels_)
    
    return df