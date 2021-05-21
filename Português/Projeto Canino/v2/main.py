"""
    Made by: Tiago Martins <ttiagommartins127@gmail.com>

    Script apenas para testar funcionalidades
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications import MobileNet
from timeit import default_timer as timer


# Dicionário com todos os modelos do tf Hub utilizados nos testes
ALL_HUB_MODELS = {
    "InceptionV3": {
        "url": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
        "size": 299
    },
    "EfficientNetB2":{
        "url": "https://tfhub.dev/google/efficientnet/b2/feature-vector/1",
        "size": 260
    },
    "EfficientNetB2_Trainable_Tf2":{
        "url": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
        "size": 260
    },
    "EfficientNetB5":{
        "url": "https://tfhub.dev/google/efficientnet/b5/feature-vector/1",
        "size": 456
    },
    "EfficientNetB5_Trainable_Tf2":{
        "url": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
        "size": 456
    },
    "MobileNetV2":{
        "url": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
        "size": 224
    },
}

ALL_TF_MODELS = {

    "MobileNet": {
        "model": MobileNet(input_shape=(224,224,3), include_top=False),
        "size": 224
    }
}



# Escolher qual modelo usar
CUR_MODEL = ALL_HUB_MODELS["EfficientNetB2_Trainable_Tf2"]

# Constante com o caminho da raiz
BASE_DIR = os.path.abspath(os.path.dirname("."))

# Definir a constante AUTOTUNE para configuração de certos métodos
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Tamanho de cada conjunto de treino/validação
BATCH_SIZE = 32

# Número de epochs
EPOCHS = 10

# Número de epochs tenhamos ficado a meio de um treino
EPOCHS_CHANGED = EPOCHS

# URL do modelo a ser utilizado pelo Hub
MODULE_HANDLE = CUR_MODEL["url"]

# Tamanho da imagem obtido através do dict do modelo
IMG_SIZE=(CUR_MODEL["size"] , CUR_MODEL["size"] )


# Booleano para saber se passámos do Warmup
WARMUP_DONE = False


class SaveModel(tf.keras.callbacks.Callback):

    def __init__(self, warmup):
        super(SaveModel, self).__init__()

        self.warmup = warmup
    
    def on_epoch_end(self, epoch, logs=None):
        
        self.model.save("saved_models/warm_up.h5")

        with open("saved_models/epochs.txt", "a+") as f:

            if epoch == 0:
                if self.warmup:
                    f.write("Warmup:\n")

                elif not self.warmup:
                    f.write("Train:\n")
                    
            f.write(str(epoch+1)+"\n")


def write_2_file(filename, content):
    """
        Função encarregue de escrever strings em um documento


        Parâmetros
        -----------
        filename: str
            Caminho do ficheiro
        
        content: str
            Conteúdo a ser escrito
    """
    
    with open(filename, "a+") as f:
        f.write(content)
    
    return

def scheduler(epoch, lr):
    """
        Função que a cada 2 epochs diminui o learning rate
        para uma aprendizagem mais lenta

        Parâmetros
        -----------
        epoch: int
            Epoch atual
        
        lr: float
            Learning rate atual

        Retorno
        --------
        return: float
            Novo learning rate
    """
    print(lr)

    if epoch % 2 == 0 and epoch != 0:
        lr *= 1./3

    return lr


def train(model, train_gen, steps, val_gen, val_steps, epochs, callback, warmup):

    """
        Função que treina o modelo

        Parâmetros
        -----------
        model: Keras Model
            Modelo criado com o Keras
        
        train_gen: Keras ImageDataGenerator
            Objeto que irá criar novas imagens com base nas fornecidas
        
        steps: int
            Número de steps de treino por epoch

        val_gen: Keras ImageDataGenerator
            Objeto que irá passar as imagens selecionadas para validação
        
        val_steps: int
            Número de steps de validação por epoch

        epochs: int
            Número de epochs
        
        callback: Keras Callback
            Objeto ou None com o callback chamado a cada epoch
        
        warmup: bool
            Boolearno que diz se é warmup ou não
        

        Retorno
        --------
        return: dict
            Dicionário com as métricas calculadas durante o treino
    """

    # Definir lista de callbacks
    callbacks = [SaveModel(warmup)]

    if callback is not None:
        callbacks.append(callback)

    # Guardar métricas calculadas durante o treino
    # na var. hist
    hist = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    return hist


# Argumentos para o gerador de imagens e para o fluxo de imagens
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")

# Gerador de imagens de validação (no caso não as transforma pois não passamos
# argumentos para tal)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

# Fluxo de imagens de validação (passará as imagens prontas para o modelo)
val_gen = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "Data"),
    subset="validation",
    shuffle=False,
    **dataflow_kwargs
)

# Gerador de imagens de treino (gera novas imagens a partir das originais = Data Aug.)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    **datagen_kwargs
)


# Fluxo de imagens de treino (passará as imagens prontas para o modelo)
train_gen = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "Data"),
    subset="training",
    shuffle=True,
    **dataflow_kwargs
)


# Verificar se é ou não para ter callback
resp = ""
callback = None

while resp != "1" and resp != "2" and resp != "3":
    resp = input("No callback/scheduler [1/2]: ")

    if resp == "2":
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


# Verificar se é para ter treino de warmup
resp = ""

while resp != "y" and resp != "n":
    resp = input("Warm up or not [y/n]: ")


# Calcular o número de steps por epoch
steps_per_epoch = train_gen.samples // train_gen.batch_size
val_steps = val_gen.samples // train_gen.batch_size




# Criar o modelo padrão

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMG_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=False),
    #tf.keras.layers.Dropout(rate),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
])


model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)


# Caso haja warm up antes do treino
if resp == "y":

    # Verificar se o Warmup ficou a meio durante algum treino
    if os.path.isfile("saved_models/epochs.txt"):
        with open("saved_models/epochs.txt", "r") as f:
            
            lines = f.read().splitlines()

            if "Train:" in lines:
                # O Warmup foi feito
                WARMUP_DONE = True
            
            # Atualizar o número de epochs
            last_epoch = lines[-1]
            EPOCHS_CHANGED = EPOCHS - int(last_epoch)
            


    ## WARM UP
    print("WARMUP BEGUN!")


    start = timer()
    
    # Se o warmup estiver feito passamos para o treino real
    if WARMUP_DONE == False:
        
        # Caso haja um modelo guardado, carregamo-lo
        if os.path.isfile("saved_models/warm_up.h5"):
            model = tf.keras.models.load_model("saved_models/warm_up.h5",
                                       custom_objects={"KerasLayer": hub.KerasLayer})

        # Treinamos o modelo
        hist = train(model, train_gen, steps_per_epoch, val_gen, val_steps, EPOCHS_CHANGED, callback, True)

        # Guardamos o modelo
        model.save("saved_models/warm_up.h5")

    end = timer()

    # Escrever o tempo demorado no ficheiro
    write_2_file("saved_models/timer.txt", "WarmUp: "+str(end-start))

    print("WARMUP ENDED\n")

    ## REAL TRAIN
    # Carregamos o modelo guardado
    model = tf.keras.models.load_model("saved_models/warm_up.h5",
                                       custom_objects={"KerasLayer": hub.KerasLayer})

    # Colocamos todas as layers do modelo como treináveis
    model.trainable = True


    # Definimos um learning rate com uma decadência exponencial em escada
    # isto é, cai exponencialmente depois que passa um dado step
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        0.01,
        decay_steps=1000,
        decay_rate=0.5,
        staircase=True
    )

    # Mudar o learning rate padrão pelo learning rate scheduler
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_scheduler),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print("NORMAL TRAIN\n")

    start = timer()

    # Se já tiver feito alguma epoch de treino real
    if WARMUP_DONE:
        EPOCHS = EPOCHS_CHANGED

    # Treinar o modelo modificado
    hist = train(model, train_gen, steps_per_epoch, val_gen, val_steps, EPOCHS, None, False)

    end = timer()

    # Escrever o tempo demorado no ficheiro
    write_2_file("saved_models/timer.txt", "Train: "+str(end-start))

    print("NORMAL TRAIN ENDED\n")


# Caso não haja warmup
else:
    print("NORMAL TRAIN")

    start = timer()

    # Treinar o modelo padrão
    hist = train(model, train_gen, steps_per_epoch, val_gen, val_steps, EPOCHS, callback)

    end = timer()

    print("NORMAL TRAIN ENDED")


# Dicionário com a média das métricas
metrics ={
    
    "acc": np.mean( hist.history["accuracy"] ), 
    "val_acc": np.mean( hist.history["val_accuracy"] ),
    "loss": np.mean( hist.history["loss"] ),
    "val_loss": np.mean( hist.history["val_loss"] )
}


# Mostrar cada métrica formatada
for k, v in metrics.items():
    print("{}: {:.3f} -".format(k, v), end=" ")

# Mostrar tempo de treino
print("\n\nExecution time: ", end-start)