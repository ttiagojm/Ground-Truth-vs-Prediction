"""
    Criar novas imagens utilizando as imagens existentes
"""
import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from time import sleep


def str_2_int(name):
    """
        Função que converte o nome das imagens
        em inteiros para que possamos ordenar elas
        com base nos números

        Parâmetros
        -----------
        name: string
            Nome da imagem
    """
    
    # Verificar se a imagem tem o sufixo _f
    # Se sim retiramos para podermos
    # prosseguir sem erros
    if "_f" in name:
        name = name.replace("_f", "")
    
    # Ficar com a parte numérica da imagem
    name = name.split(".")[0]
    
    # Converter para um inteiro
    return int(name)


def rename_last_img(path):
    """
        Adiciona um sufixo à última foto original

        Parâmetros
        -----------
        path: string
            Caminho para a pasta de com todas as classes

        Retorno
        --------
        dirs: list
            Lista com todas as classes
    """

    # Todas as pastas das classes
    dirs = os.walk(path).__next__()[1]

    for d in dirs:
        # Imagem na pasta e ordená-las de forma crescente
        # baseado nos números
        imgs = sorted(os.walk(os.path.join(path, d)).__next__()[2],
                      key=lambda f: str_2_int(f))
        
        # Verificar se já existe o sufixo
        sufix = [i for i in imgs if "_f" in i]

        if sufix:
            continue
        
        print(d, imgs)
        # Nome da última imagem
        name_last = imgs[-1]
        
        # Adicionar o sufixo _f a ela
        img_path = os.path.join(path,d,name_last)
        out_path = os.path.join(path,d,
                                name_last.split(".")[0]\
                                +"_f."+name_last.split(".")[1])
        
        os.rename(img_path, out_path)

    # Aproveitar a lista de classes
    return dirs


def brightness(img):
    """
        Função encarregue de alterar o brilho
        da imagem aleatoriamente.

        De acordo com o GeeksforGeeks
        https://www.geeksforgeeks.org/opencv-understanding-brightness-in-an-image/?ref=rp

        O brilho pode ser aumentado adicionando um viés a cada pixel
        E pode diminuido subtraindo um viés a cada pixel

        Como não se quer a imagem totalmente branca ou preta, geramos números
        no intervalo [-127, 127] sendo que número negativos serão para diminuir
        o brilho e positivos para aumentar

        Por questões de performance decidi usar uma função do OpenCV: 
        
        addWeighted(src1, alpha, src2, beta, gamma)
        
        cuja fórmula é:
        alpha * src1 + beta * src2 + gamma

        A src1 e src2 serão a mesma imagem. Como pretendo só somar
        algo à imagem irei 'anular' o primeiro produto, passando o
        alpha como 0.
        O beta passo como 1 para manter os valores da imagem e o gamma
        será então o viés/constante aleatória de brilho.


        Parâmetros
        -----------
        img: Numpy array
            Imagem original
        
        Retorno
        --------
        img: Numpy array
            Imagem transformada
    """

    betha = random.randint(0, 100)

    out_img = np.zeros(img.shape, img.dtype)

    out_img = cv2.convertScaleAbs(img, alpha=1, beta=betha)

    #img = cv2.addWeighted(img, 0, img, 1, betha)

    #img = img.astype(type_img)

    # Clipar valores para o intervalo [0, 255]
    #img = np.clip(img, 0, 255)

    return out_img


def contrast(img):
    """
        Função encarregue de ajustar o contraste aleatóriamente.

        De acordo com o GeekforGeeks:
        https://www.geeksforgeeks.org/opencv-understanding-contrast-in-an-image/

        O contraste por ser aumentado com um factor > 1 e pode ser diminuido com
        um factor < 1.

        Como não quero que o contraste seja 0, pois a imagem ficará completamente escura,
        o intervalo será [0.3, 1.5]

        Por questões de performance decidi usar uma função do OpenCV: 
        
        addWeighted(src1, alpha, src2, beta, gamma)

        cuja fórmula é:

        alpha * src1 + beta * src2 + gamma

        A src1 e src2 serão a mesma imagem. Como quero apenas que a imagem
        seja multiplicado pelo meu factor, então passarei esse factor na
        variável alpha.
        Passo 0 no beta para 'anular' o segundo produto e o gamma será 0, já
        que não quero viés adicionado.


       Parâmetros
        -----------
        img: Numpy array
            Imagem original
        
        Retorno
        --------
        img: Numpy array
            Imagem transformada
    """

    alpha = random.uniform(1.1, 3.0)

    out_img = np.zeros(img.shape, img.dtype)

    out_img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    #img = cv2.addWeighted(img, alpha, img, 0, 0)

    #img = img.astype(type_img)

    # Clipar valores para o intervalo [0, 255]
    #img = np.clip(img, 0, 255)

    return out_img


def rotate(img):
    """
        Função que faz a rotação de uma imagem usando como
        referência o centro da imagem e um ângulo aleatório.

        O ângulo é gerado num intervalo [-90, 90], em que ângulos positivos
        rotacionam em sentido anti horário.

        Primeiramente temos de obter a matriz de rotação (2x3) que depois será
        feito o produto entre ela e a imagem.

        Para fazer a rotação da imagem usamos o seguinte método:
        https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326

        Após obtermos a matriz, aplicamos à nossa imagem usando uma transformação afim:
        https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983


        Parâmetros
        -----------
        img: Numpy array
            Imagem original
        
        Retorno
        --------
        img: Numpy array
            Imagem transformada
    """

    # Centro de rotação
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # Angulo (valores positivos significam rotação no sentido anti horário)
    angle = random.randint(-90,90)
    
    # Matriz de 2 x 3 de rotaçao
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Aplicar a transformação afim
    return cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))

def saturation(img):
    """
        Função que ajusta a saturação de forma aleatória.

        Para ajustar a saturação convertemos a imagem para HSV
        para termos acesso ao canal Saturation.

        Multiplicamos todos os valores desse canal por um factor
        de intervalo [0.3, 1.5], onde valores < 1 vão diminuir a
        saturação e valores > 1 vão aumentá-la.

        Parâmetros
        -----------
        img: Numpy array
            Imagem original
        
        Retorno
        --------
        img: Numpy array
            Imagem transformada
    """
    # Precisamos converter para HSV para
    # aplicar-mos um factor no canal S (saturation)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    alpha = random.uniform(0.3, 1.5)
    
    img[...,1] = img[..., 1] * alpha

    # Clipar valores para um determinado intervalo
    # H - [0, 179]
    # S - [0, 255]
    # V - [0, 255]
    img[...,1] = np.clip(img[..., 1], 0, 255, dtype=img.dtype)

    # Voltar para BGR
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    return img

def flip_horizontal(img):
    """
        Função que espelha/reflete a imagem usando o eixo-y como
        referência.


        Parâmetros
        -----------
        img: Numpy array
            Imagem original
        
        Retorno
        --------
        img: Numpy array
            Imagem transformada
    """
    # O eixo 1 é o eixo-y, esse é o eixo
    # que usamos para fazer a nossa reflexão
    return cv2.flip(img, 1)


def gen_data(path, dirs, num_imgs=2000, spec_classes=None):
    """
        Gera novas imagens a partir das originais

        Parâmetros
        -----------
        path: string
            Caminho para a pasta de imagens
        
        dirs: list
            Lista com o nome de todas as pastas/classes
            dentro do path
        
        num_imgs: int
            Número total de imagens que queremos
        
        spec_classes: list
            Lista das classes que queremos
    """

    # Lista com todos as imagens originais
    orig = []

    # Se temos classes especificas
    # sobrescrevemos o dirs com elas
    if spec_classes:
        dirs = spec_classes
    else:
        dirs = os.walk(path).__next__()[1]

    # Colocar as funções de transformação todas numa lista
    ops = [brightness, contrast, flip_horizontal, rotate, saturation]

    for d in dirs:
        print(d)
        # Garantir que a lista de imagens
        # está vazia
        orig = []

        # Obter nome das imagens e orderná-los
        # de ordem crescente
        imgs = sorted(os.walk(os.path.join(path, d)).__next__()[2],
                      key=lambda f: str_2_int(f))

        # Indice da útlima imagem e o número da mesma
        idx_last_img = [(i, img) for i, img in enumerate(imgs) if "_f" in img]
        
        idx = int(idx_last_img[0][0])
        
        img_name = str_2_int( idx_last_img[0][1] )

        # Número de imagens
        cur_num_imgs = len(imgs)

        # Guardar imagens originais na lista orig
        orig = imgs[:idx+1]

        # Só gerar imagens se ainda não existirem
        # imagens suficientes
        if cur_num_imgs < num_imgs:

            # Recomeçar a contagem do número da última
            # imagem original.
            # Devido a querer usar o ultimo indice (img_name)
            # como sequencia para as proximas imagens
            # foi necessário calcular quantas imagens faltam
            # a partir desse indice

            end = num_imgs - cur_num_imgs

            end = (img_name+1) + end
            
            for i in tqdm(range(img_name+1, end)):
                # Selecionar uma imagem original aleatória
                img = random.choice(orig)

                # Ler imagem
                img = cv2.imread(os.path.join(path,d,img))

                # Escolher pelo menos 2 funções de transformação
                # para aplicar na imagem
                number_of_ops = random.randint(2, len(ops))

                # Escolher aleatóriamente as number_of_ops funções
                choosen_ops = random.sample(ops, number_of_ops)

                # Realizar cada operação
                for op in choosen_ops:
                    img = op(img)
                    
                # Escrever a nova imagem
                out_path = os.path.join(path, d, str(i)+".jpg")
                cv2.imwrite(out_path, img)
        else:
            print("A raça {} já tem {} imagens.".format(d, cur_num_imgs))
