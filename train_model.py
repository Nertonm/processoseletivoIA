"""
Treinamento de CNN para classificacao de digitos manuscritos (MNIST).
Saida: model.h5
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _preprocessar(x):
    """Normaliza pixels para [0,1] em float32 e adiciona a dimensao de canal."""
    if x.dtype != np.uint8:
        raise TypeError(f"Esperado uint8 (MNIST cru), recebido {x.dtype}")
    return np.expand_dims((x / 255.0).astype("float32"), axis=-1)


def carregar_dados():
    """Carrega o MNIST e aplica o pre-processamento da CNN.

    Retorna (x_train, y_train), (x_test, y_test). As imagens ficam em
    float32 no formato (N, 28, 28, 1), pronto para camadas Conv2D.
    O split de treino e teste e o padrao do proprio Keras.
    """
    print("[TRAIN] Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = _preprocessar(x_train)
    x_test = _preprocessar(x_test)

    print(f"[TRAIN] Shape treino={x_train.shape} | teste={x_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def construir_modelo():
    """Define a CNN compacta para classificacao do MNIST.

    Arquitetura:
        Input   (28, 28, 1)
        Conv2D  (32 filtros, 3x3, relu, padding=same)
        MaxPool (2x2)                                      -> 14 x 14 x 32
        Conv2D  (64 filtros, 3x3, relu, padding=same)
        MaxPool (2x2)                                      ->  7 x  7 x 64
        Flatten                                            -> 3136
        Dense   (64, relu)
        Dropout (0.3)
        Dense   (10, softmax)

    Notas de projeto:
    - padding='same' preserva a dimensao espacial apos cada Conv2D e
      mantem o fluxo 28 -> 14 -> 7 exato. Evita descarte assimetrico
      de bordas.
    - Dense(64) no lugar de Dense(128). A camada densa concentra a
      maior parte dos parametros; reduzir para 64 neuronios diminui
      o tamanho do modelo sem perda relevante de acuracia no MNIST.
    - Dropout(0.3) regulariza o modelo o suficiente para 5 epocas.

    A funcao retorna o modelo nao compilado. A compilacao fica a cargo
    da rotina de treino.
    """
    modelo = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn_edge",
    )
    return modelo
