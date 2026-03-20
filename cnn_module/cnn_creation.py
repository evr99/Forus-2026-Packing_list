from os import environ
environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.utils import image_dataset_from_directory
from keras.applications import EfficientNetB4
from keras import Model, layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import Input

from tf2onnx.convert import from_keras
from tensorflow import TensorSpec

from pathlib import Path

# Parámetros
first_train_epochs = 15
second_train_epochs = 20
num_classes = 5

data_path = Path(__file__).resolve().parent / "Train_data"
onnx_save_path = Path(__file__).resolve().parent / "onnx_models"

# Obtención del dataset
train_dataset, val_dataset = image_dataset_from_directory(
    directory = data_path,
    labels = "inferred",
    label_mode = "categorical",
    color_mode = "rgb",
    batch_size = 32,
    image_size = (224, 224),
    seed = 42,
    validation_split = 0.2,
    subset = "both"
)

# Modelo base -> Transfer learning de Efficient net.
base_model = EfficientNetB4(include_top = False, weights = "imagenet", classes = num_classes)

# Monitor del entrenamiento del modelo para early stopping.
monitor = EarlyStopping(monitor = "val_accuracy", patience = 3, restore_best_weights = True)

# Procesamiento de las imagenes -> data augmentation
data_augmentation = layers.Pipeline([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor = 0.2, interpolation = "bilinear"),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1)
], name = "data_augmentation")

# Modelo como tal, Conexión de capas
input = Input(shape = ((224, 224, 3)))
x = data_augmentation(input)
x = base_model(x, training = False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(num_classes, activation = "softmax")(x)
model = Model(input, output)

# Compilación y entrenamiento del modelo con capa congelada - Primer entrenamiento
base_model.trainable = False
model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = "categorical_crossentropy",
    metrics = ["accuracy", "mse"]
)
model.fit(
    x = train_dataset,
    validation_data = val_dataset,
    batch_size = 32,
    epochs = first_train_epochs,
    callbacks = [monitor]
)

# Compilación y entrenamiento del modelo con capa descongelada - Segundo entrenamiento
base_model.trainable = True
model.compile(
    optimizer = Adam(learning_rate = 0.00001),
    loss = "categorical_crossentropy",
    metrics = ["accuracy", "mse"]
)
model.fit(
    x = train_dataset,
    validation_data = val_dataset,
    batch_size = 32,
    epochs = second_train_epochs,
    callbacks = [monitor]
)

# Guardar el modelo bajo la métrica obtenida
acc_model = str(model.get_metrics_result()["accuracy"])[0:5]
mse_model = str(model.get_metrics_result()["mse"])[0:5]
onnx_save_path = f"{onnx_save_path}/{acc_model}_{mse_model}.onnx"

# guardar .onnx
model_proto, _ = from_keras(
    model,
    input_signature = [
        TensorSpec(model.inputs[0].shape,dtype = model.inputs[0].dtype,name = model.inputs[0].name)
        ],
    output_path = onnx_save_path,
)

# Name: tensorflow-intel tensorflow
# Version: 2.16.1 -> 2.17.0


# img2table 1.4.2 requires pypdfium2==4.30.0, but you have pypdfium2 5.3.0 which is incompatible.
# paddlepaddle 3.3.0 requires opt-einsum==3.3.0, but you have opt-einsum 3.4.0 which is incompatible.
# streamlit-extras 0.7.8 requires protobuf>=5.27.3, but you have protobuf 3.20.3 which is incompatible.