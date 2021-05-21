"""
    Script para analisar o feature_breeds e poder entender que raças
    se relacionam ou que nao se relacionam
"""
from cluster import get_clusters
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import cv2
import os
import sys
sys.path.append("..")
from utils import resize_img

# Caminho para o CSV
DIR = os.path.abspath(os.path.dirname("."))
DIR_FEAT = os.path.join(DIR, "feature_breeds.csv")

# Verificar se já foi criado o cvs com grupos
isGroup = os.path.isfile("grouped_feature_breeds.csv")

if isGroup:
    DIR_GROUPED = os.path.join(DIR, "grouped_feature_breeds.csv")


def find_breed_by_group(breed, df):
    """
        Recebe uma raça e procura o seu grupo no DataFrame

        breed: string
            Raça

        df: DataFrame
            DataFrame com todas as caracteristicas e grupos de
            cada raça
    """

    print(df[df["Raca"] == breed])


def create_df(dir_path, isOne_Hot):
    """
        Função encarregue de criar e retornar um DataFrame
    """

    # Carregar o CSV
    df = pd.read_csv(dir_path, encoding="latin-1")

    # Guardar a Series com os nomes das raças
    names = df["Raca"]

    # Remover Coluna Raca e URL
    df.drop(["Raca"], axis=1, inplace=True)

    # Converter todas as features (exceto as raças) para one hot
    if isOne_Hot:
        df.drop(["URL"], axis=1, inplace=True)
        df_oneh, cat = one_hot(df)

    # Converter todas as features (exceto as raças) para numeros inteiros
    else:
        df_oneh, cat = sparse_encoding(df)

    # Concatenar o novo DataFrame one hot com as raças
    df = pd.concat([names, df_oneh, ], axis=1)

    # Concatenar as raças ao DataFrame original novamente
    cat = pd.concat([names, cat], axis=1)

    return df, cat


def set_text_img(img, label):
    """
        Coloca um texto em uma imagem

        img: Numpy Array
            Imagem original

        label: string
            Texto para colocar na imagem (no caso é a raça)

        return: Numpy Array
            Imagem com texto embutido
    """
    # Tamanho da imagem e posição do texto
    h, w = img.shape[0], img.shape[1]
    x, y = 0, 20

    rect = img.copy()
    orig = img.copy()

    # Adicionar retângulo preto
    cv2.rectangle(rect, (x, h-40), (w, h), (0, 0, 0), -1)

    # Adicionar texto ao retangulo
    cv2.putText(rect, label, (x, h-10),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.7, (255, 255, 255), 1,
                cv2.LINE_AA)

    # Adicionar o texto com rectangulo à imagem
    return cv2.addWeighted(rect, 0.6, orig, 1-0.6, 0)


def one_hot(df):
    """
        Transforma as colunas em one hot

        df: DataFrame
            DataFrame com todas as raças e suas caracteristicas

        Retorno:

        DataFrame
            Dataframe com todas as features transformadas em one hot

        cat: DataFrame
            Dataframe com todas as features não transformadas 
            (preserva o Dataframe original)
    """
    cols = []
    for col in df.columns:
        if not col == "Raca":
            cols.append(col)

    # Preservar o DataFrame com as variáveis categóricas
    cat = df.copy()

    return pd.get_dummies(df[cols], drop_first=True), cat


def sparse_encoding(df):
    """
        Transforma as colunas em número inteiros

        df: DataFrame
            DataFrame com todas features incluindo
            os grupos definidos pelo clustering
    """

    # Nome das colunas para fazer encoding
    cols = []
    for col in df.columns:
        if not col == "Raca":
            cols.append(col)

    # DataFrame original
    cat = df.copy()

    # Objeto para o encoding
    le = LabelEncoder()

    # Transformar cada coluna
    for col in cols:
        df[col] = le.fit_transform(df[col].to_numpy())

    return df, cat


def add_weights(features, cols):
    """
        Aplica pesos aos valores das features.

        features: DataFrame
            DataFrame com todas as features

        cols: dict
            Dicionário com os nomes das colunas de cada caracteristica geral
            Ex: Focinho: [Focinho_curto, Focinho_normal, Focinho_longo] 

        return: DataFrame
            DataFrame com os pesos aplicados
    """

    # Para os valores == 0, poderem sofrer alterações com pesos
    epsilon = 0.0001

    # Caracteristicas gerais do cão e o respetivo peso
    main_char = {
        "Tamanho": 0.8,
        "Cor": 0.2,
        "Focinho": 0.7,
        "Orelhas": 0.7,
        "Pelo": 0.5
    }

    for k, v in main_char.items():

        # Identificar todas as colunas que representam
        # a caracteristica geral
        main_char_cols = [col for col in cols if k in col]

        # Aplicar o peso
        df[main_char_cols] = df[main_char_cols].apply(lambda x: (epsilon+x)*v)

    return df


def cosine_similarity(matrix):
    """
        Calcula o cosseno de similiaridade entre cada raça
    """
    # Constante usada para prevenir raizes de 0
    epsilon = 0.001

    # Raças
    names = matrix[:, 0]

    # Manter apenas as features
    features = matrix[:, 1:]

    # Criar um array só para as similiares
    sim = np.zeros((matrix.shape[0], matrix.shape[0]))

    # Interagir com cada linha da matriz de features
    for i, row in enumerate(features):
        for j, row2 in enumerate(features):
            # Produto escalar entre vetores
            dot_prod = np.dot(row, row2) + epsilon

            # Distancia entre os vetores
            dist = np.sqrt(np.sum(row**2 + epsilon)) * \
                np.sqrt(np.sum(row2**2 + epsilon))

            # Guardar o resultado da divisão
            sim[i, j] = dot_prod/dist

    # Criar um dicionário com todos os nomes e inicializar com zeros
    names_counter = dict.fromkeys(names, 0)

    # Vamos passar por todas as linhas da matriz de similaridade
    for i, row in enumerate(sim):

        # Ordenar por ordem crescente e obter o menor (mais distinta raça em comparação com a raça i)
        min_vals = np.argsort(row)[:1]

        # Converter os indices nos nomes das raças
        min_vals = [names[idx] for idx in min_vals]

        # Incrementar a raça mais distinta
        names_counter[min_vals[0]] += 1

        #print("Raça: {} - Raças mais distintas: {}".format(names[i], min_vals))

    # Ordenar dict por maior valor
    names_counter = dict(sorted(names_counter.items(),
                                key=lambda item: item[1], reverse=True))

    return names_counter


# Criar um loop para escolher qual método
# queremos para calcular as diferentes raças
resp = ""

while resp != "y" and resp != "n":

    resp = input(
        "[*] Quer usar os grupos criado pelo alg. de clustering? [y/n]:").lower()

    if resp == "n" or not isGroup:

        # Criar o DataFrame com as caracteristicas das nossas raças
        df, cat = create_df(DIR_FEAT, True)

        # Aplicar pesos
        df = add_weights(df, df.columns.to_numpy())

    elif resp == "y" and isGroup:
        # Criar o DataFrame com as caracteristicas das nossas raças
        # e os grupos
        df, cat = create_df(DIR_GROUPED, False)

# Calcular a similaridade entre os cossenos de cada raça e as restantes
diff_breeds = cosine_similarity(df.to_numpy())

print(diff_breeds)

# Plotar um gráfico com as raças usadas
num_breeds = 5

# Caminho para o dataset completo
DIR_DATASET = os.path.abspath('../')
DIR_DATASET = os.path.join(DIR_DATASET, "Data")

# Obter apenas as primeiras num_breeds
most_diff = list(diff_breeds.keys())[:num_breeds]

# Obter um DataFrame com as classes separadas por possíveis grupos
df_groups = get_clusters(df, cat, 16, prune=5)

for breed in most_diff:
    # print(breed)

    # Caminho para a primeira imagem de cada raça
    DIR_BREED = os.path.join(DIR_DATASET, breed)

    # print(DIR_BREED)

    img = os.walk(DIR_BREED).__next__()[2]
    img = os.path.join(DIR_BREED, img[0])

    # Carregar a imagem
    img = cv2.imread(img)

    # Redimensionar a imagem
    img = resize_img(img, 229, aspect_ratio=True)

    # Obter o grupo da respetiva raça
    # se df_groups tiver algo
    if not df_groups.empty:
        find_breed_by_group(breed, df_groups)

        # Ordenar DataFrame por grupo
        df_groups = df_groups.sort_values(by=["Grupo"])

        # Guardar DataFrame com grupos num csv
        df_groups.to_csv("grouped_feature_breeds.csv", index=False)

    # Colocar o nome da raça na imagem
    img = set_text_img(img, breed)

    cv2.imshow(breed, img)


# Manter a janela aberta e destruir após
# pressionar tecla
cv2.waitKey(0)
cv2.destroyAllWindows()
