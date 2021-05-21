"""
    Script que controla o fluxo principal do modelo
"""
from model import train
from gen_data import rename_last_img, gen_data
from download_data import download


# Classes que queremos usar para treinar
##classes = ["Afghan Hound", "Kelpie", "Irish Water Spaniel", "Old English Sheepdog"]

# Download de imagens
##download()

# Colocar sufixo na última imagem de cada classe
dirs = rename_last_img("Data")

# Gerar novas imagens
#gen_data("Data", dirs)

# Treinar
train(freeze_layers=True, val_percent=0.2)