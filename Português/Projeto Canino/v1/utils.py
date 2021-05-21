"""
    Funções de auxilio
"""
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import PIL
import requests
import shutil
import os
import cv2
import threading
import pandas as pd


def threaded_job(job):
    # Iniciar a thread e executar
    thread = threading.Thread(target=job)
    thread.start()


def resize_img(img, size, aspect_ratio=True):
    """
        Função para redimensionar uma imagem (numpy array)
        para size x size

        img: Numpy array
            Imagem para ser redimensionada
        
        size: Tamanho da imagem, onde H = W
        
        aspect_ratio: bool
            Mantém ou não o aspect ratio

        return: Numpy array
            Imagem redimensionada
    """

    # Altura e comprimento
    H, W = img.shape[0], img.shape[1]

    # Interpolação
    inter = cv2.INTER_AREA

    # Se não for para manter o aspect ratio
    # ou se as imagens forem quadradas
    # fazemos um redimensionamento normal
    if not aspect_ratio or img.shape[0] == img.shape[1]:
        return cv2.resize(img, (size, size), interpolation=inter)

    # Deltas de redimensionamento (lembrando que o delta da dimensão mais pequena
    # é 0)
    dW = 0
    dH = 0

    # Se a altura for maior que o comprimento
    # então começamos por redimensionar o comprimento
    # para o tamanho desejado e escalonamos(rescale) a altura
    if H > W:
        # Rescaling da altura
        r = size / float(W)
        H = int(H * r)

        # Comprimento que queremos
        W = size
        img = cv2.resize(img, (W, H), interpolation=inter)
        
        # Redimensionar a altura
        dH = int((H - size) / 2.)

    else:
        # Rescaling do comprimento
        r = size / float(H)
        W = int(W * r)

        # Altura que queremos
        H = size
        img = cv2.resize(img, (W, H), interpolation=inter)

        # Redimensionar o comprimento
        dW = int((W - size) / 2.)

    
    # Obter a parte da imagem que pretendemos
    img= img[dH: H - dH, dW: W - dW]

    return cv2.resize(img, (size, size), interpolation=inter)

def get_breed_url(breed):
    """
        Lê o .csv das raças e obtém o url
        da raça passado por parâmetro

        breed: string
            Raça para obter url
        
        return: string
            Url da raça desejada
    """
    # Carregar o .csv para um DataFrame
    df = pd.read_csv("analysis/feature_breeds.csv")

    # Lista com o URL
    url = df[df["Raca"] == breed]["URL"].values

    # Se lista estiver vazia é porque é uma raça
    # selvagem e nós não a queremos
    if url:
        url = url[0]
        return url

    return None

def verify_img(path):
    """
        Função que verifica se uma imagem está num formato
        conhecido, caso contrário será eliminada

        path: string
            Caminho para a imagem
        
        return: bool
            Caso seja um formato conhecido retorna True, se não retorna False
    """

    # Testar se a imagem é possivel ser carregada
    try:
        img = Image.open(path)

    # Se não for possivel, será eliminada
    except UnidentifiedImageError:
        os.remove(path)
        return False
    
    return True

def get_imagenet(url, path):
    """
        Função que descarrega todas as imagens presentes no URL
        e guarda no path

        url: string
            URL onde estarão os links das imagens
        
        path: string
            Caminho para a pasta onde ficarão guardadas as imagens
    """

    # Obter o html da página referente ao url 
    html = requests.get(url).content

    # Guardar link linha por linha
    links = html.decode("utf-8").split("\r\n")
    
    # Remover o último \n
    links = links[:-1]


    # Loop por todos os links para podermos fazer download
    # Usarei o tqdm para ter uma noção do progresso de download
    for i in tqdm(range(len(links))):

        link = links[i]

        # Alguns links não são mais acessiveis, logo
        # o except irá retornar o erro e passar para
        # a proxima imagem
        try:
            
            # Download imagem (stream)
            req = requests.get(link, stream=True, timeout=30)

            # Se a imagem existir continuar
            if req.status_code == 200:
                
                # Tentar guardar
                try:
                    img_path = os.path.join(path, str(i)+".jpg")

                    # Verificar se a imagem já existe
                    # se sim passamos à próxima
                    if os.path.exists(img_path):
                        continue
                    
                    # Escrevemos a imagem
                    with open(img_path, "wb") as f:
                        req.raw.decode_content = True
                        shutil.copyfileobj(req.raw, f)


                    # Se for realmente uma imagem JPG, mantém-se guardada
                    if verify_img(img_path):
                        print("[+] Imagem guardada com sucesso!")
                    
                    # Caso contrário, será eliminada
                    else:
                        print("[*] Imagem eliminada por ter um formato desconhecido")

                # Caso não consiga guardar mostrar mensagem
                except:
                    # Caso ele tenha escrito a imagem
                    # eliminamos
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    
                    print("[!] Não conseguiu guardar a imagem: {}".format(link))

            # Caso não exista apenas mostra uma mensagem
            else:
                print("[-] Link não encontrado")
        
        except (requests.exceptions.ConnectionError,
                requests.exceptions.TooManyRedirects,
                requests.exceptions.ReadTimeout,
                requests.exceptions.InvalidURL) as e:
            print("[!] Erro de Conexão: ", e)
            continue
    
    # Limpar terminal
    os.system("cls" if os.name=="nt" else "clear")