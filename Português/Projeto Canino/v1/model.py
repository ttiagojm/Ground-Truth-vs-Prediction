"""
    Script onde é criado o modelo, treino e validação
"""

from tensorflow.keras.applications.inception_v3 import InceptionV3 # Modelo para Transfer Learning
#from models.inceptionV3 import inception_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime # Colocar datas no nome dos dados escritos
import tensorflow as tf
import numpy as np # Para ordenar as probs por ordem crescente
import os
import math
#import matplotlib.pyplot as plt
# %matplotlib inline


# Para redimensionar as imagens
IMG_H = 199
IMG_W = 199

# Número de imagens por lote/batch
BATCH_SIZE = 32

# Epochs
EPOCHS = 10

# Caminho do diretorio deste script
DIR = os.path.abspath(os.path.dirname("."))

# Caminho para o dataset de treino
DIR_TRAIN_DS = "Data"


def prettify_classnames(path):
    """
      Função que transforma uma lista de nomes, os quais podem estar separados por 
      hífens e/ou underlines, em nomes com letra maiúscula e separados com espaços.


      Parâmetros
      -----------
      path: string
        Diretorio das pastas (classes) a serem preprocessados
    """

    class_names = os.walk(path).__next__()[1]

    for class_name in class_names:

        # Guardar o nome antes de ser transformado
        old_class_name = class_name

        # Se não tiver underline, é um nome apenas
        if "_" not in class_name:

            # Retirar a primeira letra e torná-la maiúscula
            first_letter = class_name[0].upper()

            # Remover a primeira letra
            class_name = class_name[1:]

            # Juntamos a letra maiuscula com o restante
            class_name = first_letter + class_name

        # Caso tenha underline
        else:

            # Verificar se tem hífen, caso sim substituir por espaço
            if "-" in class_name:
                class_name = class_name.replace("-", " ")

            # Subsitutir underline por espaço
            class_name = class_name.replace("_", " ")

            # Dividir strings por espaço
            split_string = ""

            for string in class_name.split(" "):
                # Retirar a primeira letra e torná-la maiúscula
                first_letter = string[0].upper()

                # Remover a primeira letra
                string = string[1:]

                # Juntamos a letra maiuscula com o restante
                string = first_letter + string + " "

                split_string += string

            # Remover espaços extras
            class_name = split_string.strip()

        old_dir = os.path.join(path, old_class_name)
        new_dir = os.path.join(path, class_name)

        os.rename(old_dir, new_dir)

def preprocess_img(img_path, label):
    """
      Função encarregue de ler as imagens de um dado caminho, redimensioná-las
      e retorná-las em Tensor em conjunto com a sua classe numérica

      Parâmetros
      -----------
      img_path: Tensor (tf.string)
        Tensor 1D com o caminho da imagem

      label: Tensor (tf.int32)
        Tensor 1D com a classe numérica da imagem


      Retorno
      ---------
      return: tuple
        Tupla com um Tensor 3D (imagem) e um Tensor 1D com a classe numérica
    """

    # Ler ficheiro
    img = tf.io.read_file(img_path)

    # Decodificar o ficheiro jpeg
    img = tf.image.decode_jpeg(img, channels=3)

    # Redimensionar para o shape pretendido
    img = tf.image.resize(img, [IMG_W, IMG_H])

    # Normalizar pixeis, intervalo = [0,1]
    img *= 1./255

    return (img, label)


def prepare_dataset(ds, buffer_size, batch_size):
    """
      Função que aplica funções de otimização de I/O como cache e prefetch
      e transforma o Dataset em uma Dataset dividido por lotes/batches

      Parâmetros
      -----------
      ds: Dataset
        Dataset com as imagens(Tensor 3D) e as labels (Tensor 1D)

      buffer_size: int32
        Tamanho do buffer (usado para fazer o shuffle)

      batch_size: int32
        Tamanho de cada lote (número de imagens/labels por batch)


      Retorno
      --------
      ds: Dataset
        Dataset processado e dividido em batches
    """
    ds = ds.shuffle(buffer_size).batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def read_images(path, batch_size, val_split, spec_classes=None):
    """
      Função que vai preprocessar cada imagem e vai adicioná-las
      aos datasets de treino e avaliação, já devidamente repartidos


      Parâmetros
      -----------
      path: string
        Caminho para o diretório com o conjunto de pastas (1 por raça)

      batch_size: int32
        Tamanho de cada lote (número de imagens/labels por batch)

      val_split: float
        Quantidade de samples para validação

      spec_classes: list
        Lista apenas com as classes que queremos usar

      Retorno
      --------
      train_ds: Dataset
        Dataset processado e dividido para treinar o modelo

      val_ds: Dataset
        Dataset processado e dividido para avaliar o modelo

      prettify_classnames(classes): list
        Lista com as classes preprocessadas de modo a que sejam
        mais legíveis
    """

    img_paths, labels = [], []

    # Serve para criar labels de tipo int
    label = 0

    # Obter todas as classes com base nos nomes de todas as pastas no diretorio
    classes = sorted(os.walk(path).__next__()[1])

    # Verificar se há classes especificas que queremos usar
    if spec_classes is not None:
        classes = [c for c in classes if c in spec_classes]

    # Loop em cada pasta/label/classe
    for c in classes:
        c_dir = os.path.join(path, c)

        # Vai andar pelo diretório da classe atual
        walk = os.walk(c_dir).__next__()

        # Extrair todos os ficheiros encontrados
        for sample in walk[2]:
            if sample.endswith(".jpg") or sample.endswith(".jpeg"):
                img_paths.append(os.path.join(c_dir, sample))
                labels.append(label)

        # Próxima classe numérica
        label += 1

    # Converter tudo em Tensors
    img_paths = tf.convert_to_tensor(img_paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Converter Tensor de labels em uma matriz one-hot
    labels_oh = tf.one_hot(labels, len(classes))

    # Criar um dataset com os paths e labels
    images = tf.data.Dataset.from_tensor_slices((img_paths, labels_oh))

    # Criar um novo dataset preprocessado
    pre_ds = images.map(preprocess_img,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Número de samples de treino e teste
    n_test_samples = math.floor(labels_oh.shape[0] * val_split)
    n_train_samples = labels_oh.shape[0] - n_test_samples

    print("Samples Treino: {:d} - Samples Avaliação: {:d}".format(
        n_train_samples, n_test_samples))

    # Criar um dataset de treino
    train_ds = prepare_dataset(
        ds.take(n_train_samples), n_train_samples, batch_size)

    # Criar um dataset de avaliação
    val_ds = prepare_dataset(ds.skip(n_test_samples),
                             n_test_samples, batch_size)

    return train_ds, val_ds, classes



def create_callbacks():

    """
        Função que cria todos os objetos para escrever as métricas e o gradientes
        no disco e, posteriormente, serão plotados no Tensorboard.

        Retorno
        --------
        summaries: dict
            Dicionário cujas keys coincidem com o nome das funções de acurácia
            e custo e os valores são os objetos para escrita
        
        grad_writer: tf.summary.writer
            Objeto para escrever os gradientes em disco
    """

    # Acuracia de treino
    acc_dir = os.path.join(DIR, "logs", "acc", "acc_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(acc_dir, exist_ok=True)

    # Acuracia de validação
    val_acc_dir = os.path.join(DIR, "logs", "val_acc", "val_acc_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(val_acc_dir, exist_ok=True)

    # Custo de treino
    loss_dir = os.path.join(DIR, "logs", "loss", "loss_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(loss_dir, exist_ok=True)

    # Custo de validacao
    val_loss_dir = os.path.join(DIR, "logs", "val_loss", "val_loss_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(val_loss_dir, exist_ok=True)

    # Gradientes
    grad_dir = os.path.join(DIR, "logs", "gred", "gred_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(grad_dir, exist_ok=True)

    ## Criar summaries para todos
    acc_sum_writer = tf.summary.create_file_writer(acc_dir)
    val_acc_sum_writer = tf.summary.create_file_writer(val_acc_dir)
    loss_sum_writer = tf.summary.create_file_writer(loss_dir)
    val_loss_sum_writer = tf.summary.create_file_writer(val_loss_dir)
    grad_sum_writer = tf.summary.create_file_writer(grad_dir)

    # Dicionário com o summaries indexados por keys
    summaries = {
        "acc": acc_sum_writer,
        "val_acc": val_acc_sum_writer,
        "loss": loss_sum_writer,
        "val_loss": val_loss_sum_writer
    }
    
    return summaries, grad_sum_writer


def create_model(num_classes, freeze_layers=False):

    """
        Função para criar o modelo. Usaremos o InceptionV3 como modelo para
        transfer learning já que o nosso dataset carece de imagens.

        Parâmetros
        -----------

        num_classes: int
            Número de classes/labels únicos
        
        input_shape: tuple
            Tupla com o tamanho do Input

        freeze_layers: bool
            Booleano que diz se é para congelar os pesos do modelo pré-treinado
            (InceptionV3), ou seja, para não treinar mais esses pesos ou se é
            para continuar o treino dos pesos
    """

    # Criar o modelo de transfer learning
    # Não incluir o topo significa não incluir as layers High-Level, ou seja,
    # as últimas layers que passam as informações para a softmax
    
    base_inception = InceptionV3(
        weights="imagenet", include_top=False, input_shape=(IMG_W, IMG_H, 3))
    
    #base_inception = inception_model(shape=(IMG_H, IMG_W, 3))

    
    # High-Level layers criadas para o nosso problema específico (classificar raças)
    out = base_inception.output
    #out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Flatten()(out)
    print(out.shape)
    #out = tf.keras.layers.Dropout(0.5)(out)
    #out = tf.keras.layers.Dense(256, activation="relu")(out)
    predict = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    print(base_inception.input)
    # Criar o modelo
    model = tf.keras.Model(inputs=base_inception.input, outputs=predict)

    
    # Congelar as layers do InceptionV3, para que não sejam treinados os parâmetros
    for layer in base_inception.layers:
        # Se freeze_layers for True, então não vamos querer que haja
        # variáveis treináveis, por isso temos de negar
        layer.trainable = not freeze_layers

    return model


@tf.function
def train_step(model,optimizer, loss_fn,
               X_train, y_train, metrics):
    
    """
        Função executada em cada batch do Dataset, encarregue de calcular
        gradientes, métricas e atualizar os parâmetros da rede.

        Parâmetros
        -----------

        model: Keras Model
            O modelo gerado pela nossa função create_model()
        
        optimizer: Keras optimizer
            Otimizador do keras para seguir a melhor direção dos
            gradientes

        loss_fn: Keras loss function
            Função de custo do keras que será derivada para obter
            os gradientes e atualizar os pesos

        X_train: Tensforflow Dataset
            Dataset com um batch de tamanho BATCH_SIZE que tem
            as imagens de input
        
        y_train: Tensforflow Dataset
            Dataset com um batch de tamanho BATCH_SIZE que tem
            os labels de cada input
        
        metrics: list
            Lista com todas funções de métricas como a acurácia e 
            o custo 

        Retorno
        --------
        grads: Tensor
            Tensor com todos os gradientes por imagem de input
            também chamado de Tensor de jacobianas que nada mais
            é que uma matriz com todas as derivadas parciais de
            primeira ordem das funções derivadas
    
    """
    
    # Monitorar as funções executadas
    # para depois podermos obter os gradientes
    with tf.GradientTape() as tape:

        # Obter as previsões do modelo
        probs = model(X_train, training=True)

        # Calcular o custo das previsões erradas
        loss_value = loss_fn(y_train, probs)
    
    
    # Calcular acurácia neste batch
    for obj in metrics:
        if obj.name == "acc":
            obj.update_state(y_train, probs)
    
    # Calcular e guardar os gradientes de todos os
    # parâmetros treináveis
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Aplicar os gradientes a esses parâmetros treináveis
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Atualizar métricas
    for obj in metrics:
        obj(y_train, probs)


    return grads


@tf.function
def test_step(model, X_test, y_test, metrics):
    val_probs = model(X_test, training=False)

    """
        Função executada em cada batch do Dataset, encarregue de avaliar
        o nosso modelo e atualizar métricas
    """

    # Calcular acurácia neste batch
    for obj in metrics:
        if obj.name == "acc":
            obj.update_state(y_test, val_probs)
    
    # Atualizar métricas
    for obj in metrics:
        obj(y_test, val_probs)


def train(freeze_layers, val_percent=0.2, spec_classes=None, optimization_alg="sgd"):

    """
        Funçõe encarregue de toda a lógica de treino do modelo, desde configurar
        os datasets de treino e avaliação até às previsões

        Parâmetros
        -----------
        freeze_layers: bool
            Se True, então as layers do Inception serão congeladas (não treinadas).
            Caso contrário não serão congeladas (serão treinadas)
        
        val_precent: float
            Percentagem de dados para avaliação
        
        spec_classes: list
            Se não for None, será uma lista com os nomes das raças de cães
            que queremos que o modelo classifique/distinga
        
        optimization_alg: string
            Otimizador a ser utilizado (por defeito, será o SGD)
    """

    ## Descomentar caso se queria alterar o nome das pastas para ficarem 
    ## mais legiveis 
    # prettify_classnames(DIR_TRAIN_DS)

    ## Carregar os Datasets das imagens e as classes
    train_ds, val_ds, classes = read_images(DIR_TRAIN_DS,
                                            BATCH_SIZE,
                                            val_percent,
                                            spec_classes)

    ## Obter o modelo
    model = create_model(len(classes), freeze_layers=freeze_layers)


    ## Definir otimizador, funçao custo e metricas
    
    # Otimizador
    if optimization_alg == "sgd":
        optimizer = tf.keras.optimizers.SGD()
    
    elif optimization_alg == "adam":
        optimizer = tf.keras.optimizers.Adam()


    # Função custo
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Acurácias de treino e de avaliação
    acc = tf.keras.metrics.CategoricalAccuracy("acc")
    val_acc = tf.keras.metrics.CategoricalAccuracy("val_acc")

    # Custo de treino e de avaliação
    loss = tf.keras.metrics.CategoricalCrossentropy("loss", dtype=tf.float32)
    val_loss = tf.keras.metrics.CategoricalCrossentropy("val_loss", dtype=tf.float32)


    # Listas com as métricas de treino e avaliação
    train_metrics = [acc, loss]
    test_metrics = [val_acc, val_loss]

    # Retornar os objetos para, posteriormente, 
    # podermos escrever as metricas 
    summaries, grad_writer = create_callbacks()

    ## Verificar se pesos existem
    ## se sim vamos aplicar ao nosso modelo
    checkpoint_path = "checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("[*] Carregar pesos")
    
    ## Guardar métricas para no fim fazermos uma média
    acc_metric = np.zeros(EPOCHS, dtype=np.float)
    val_acc_metric = np.zeros(EPOCHS, dtype=np.float)
    loss_metric = np.zeros(EPOCHS, dtype=np.float)
    val_loss_metric = np.zeros(EPOCHS, dtype=np.float)

    ## Loop de treino e avaliação
    for epoch in range(EPOCHS):

        # Definir qual o step/epoch em que estamos
        # para escrever nos gráficos do Tensorboard
        tf.summary.experimental.set_step(epoch)


        # Treinar
        for step, (x_batch, y_batch) in enumerate(train_ds):
            grads = train_step(model, optimizer, loss_fn,
                            x_batch, y_batch, train_metrics)
        
        
        # Avaliar
        for x_batch, y_batch in val_ds:
            test_step(model, x_batch, y_batch, test_metrics)

        # Guardar checkpoint
        manager.save()

        # Mostrar progresso
        print("Epoch {} - Loss: {:.3f} Val_Loss: {:.3f} - Acc: {:.3f} - Val_Acc: {:.3f}".format(
        (epoch+1), 
        loss.result(),
        val_loss.result(), 
        acc.result(),
        val_acc.result()))

        # Guardar os resultados
        loss_metric[epoch] = loss.result()
        val_loss_metric[epoch] = val_loss.result()
        acc_metric[epoch] = acc.result()
        val_acc_metric[epoch] = val_acc.result()

        # Guardar as métricas
        for obj in (train_metrics+test_metrics):
            with summaries[obj.name].as_default():
                tf.summary.scalar(obj.name, obj.result())
            

            # Resetar o estado no fim de atualizar
            obj.reset_states()

            summaries[obj.name].flush()

        # Escrever os gradientes
        with grad_writer.as_default():
            for i, g in enumerate(grads):
                
                # Calcular média de gradientes
                mean = tf.reduce_mean(tf.abs(g))

                tf.summary.scalar("gradient_mean_layer_{}".format(i+1), mean)
                tf.summary.histogram("gradient_hist_layer_{}".format(i+1), g)


                # Limpar o buffer após escrever
                grad_writer.flush()
    
    ## Fazer médias e escrever num ficheiro
    with open("metric_mean.txt", "a+") as f:
        f.write("model - epochs:{} - batch: {} - size: {} - opt: {} - val{}: trainable: {}\n"\
            .format(EPOCHS, BATCH_SIZE, IMG_H, optimization_alg,val_percent, not freeze_layers))
        
        f.write("Acc: {}\n".format(np.mean(acc_metric)))
        f.write("Val_Acc: {}\n".format(np.mean(val_acc_metric)))
        f.write("Loss: {}\n".format(np.mean(loss_metric)))
        f.write("Val_Loss: {}\n\n".format(np.mean(val_loss_metric)))


#k_top = 3
# for i in np.flip(np.argsort(pred)[0, -k_top:]):
#    print("Raça: {:s} - Prob: {:.2f}".format(classes[i], pred[0, i] * 100))
