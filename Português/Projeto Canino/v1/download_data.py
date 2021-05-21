import os
from utils import get_imagenet, get_breed_url

DIR = os.path.abspath(os.path.dirname("."))

def verify_downloaded_imgs(breed):
    """
        Verifica se já existe uma pasta com imagens
        de uma dada raça

        breed: string
            Raça a ser verificada

        return: bool, path
            True caso já exista, se não False
            Enviamos o path porque iremos precisar dele
    """
    path = os.path.join(DIR, "Data", breed)

    return os.path.exists(path), path


def download():

    # Ler as todas as raças escritas num ficheiro
    with open("breeds.txt", "rb") as f:
        for breed in f:
            # Converter os bytes para string e remover
            # as quebras de linhas
            breed = breed.decode("utf-8").replace("\n", "")

            # Verificar se a raça já foi descarregada
            isdownloaded, path = verify_downloaded_imgs(breed)

            # Se ainda não foi, fazemos o download
            if not isdownloaded:
                print("A descarregar a raça: {}".format(breed))
                # Criar a respetiva pasta
                os.makedirs(path, exist_ok=True)

                # Obter o URL
                url = get_breed_url(breed)

                # Se retornar None é porque é uma raça selvagem
                # e não queremos essa raças
                if url == None:
                    continue

                # Fazer o download das imagens
                get_imagenet(url, path)
            
            else:
                print("A raça {} já foi descarregada.".format(breed))