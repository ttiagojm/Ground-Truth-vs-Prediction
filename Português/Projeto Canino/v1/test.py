"""
    Script apenas para testar funcionalidades
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from timeit import default_timer as timer

ALL_MODELS = {
    "InceptionV3": {
        "url": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
        "size": 299
    },
    "EfficientNetB2":{
        "url": "https://tfhub.dev/google/efficientnet/b2/feature-vector/1",
        "size": 260
    },
    "EfficientNetB5":{
        "url": "https://tfhub.dev/google/efficientnet/b5/feature-vector/1",
        "size": 456
    },
    "MobileNetV2":{
        "url": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
        "size": 224
    },
}

CUR_MODEL = ALL_MODELS["InceptionV3"]
BASE_DIR = os.path.abspath(os.path.dirname("."))
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE=(CUR_MODEL["size"] , CUR_MODEL["size"] )
MODULE_HANDLE = CUR_MODEL["url"]


datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

val_gen = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "Data"),
    subset="validation",
    shuffle=False,
    **dataflow_kwargs
)


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    **datagen_kwargs
)

train_gen = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "Data"),
    subset="training",
    shuffle=True,
    **dataflow_kwargs
)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMG_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=False),
    #tf.keras.layers.Dropout(rate),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

steps_per_epoch = train_gen.samples // train_gen.batch_size
val_steps = val_gen.samples // train_gen.batch_size

start = timer()

hist = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps
)

end = timer()


metrics ={
    
    "acc": np.mean( hist.history["accuracy"] ), 
    "val_acc": np.mean( hist.history["val_accuracy"] ),
    "loss": np.mean( hist.history["loss"] ),
    "val_loss": np.mean( hist.history["val_loss"] )
}

for k, v in metrics.items():
    print("{}: {:.3f} -".format(k, v), end=" ")

print("\n\nExecution time: ", end-start)