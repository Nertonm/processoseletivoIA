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

    Retorna (x_train, y_train), (x_test, y_test), com x em float32 no
    formato (N, 28, 28, 1) pronto para alimentar camadas Conv2D.
    O split train/test e o fornecido pelo proprio Keras (deterministico).
    """
    print("[TRAIN] Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = _preprocessar(x_train)
    x_test = _preprocessar(x_test)

    print(f"[TRAIN] Shape treino={x_train.shape} | teste={x_test.shape}")

    return (x_train, y_train), (x_test, y_test)
