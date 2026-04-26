"""
Treinamento de CNN para classificacao de digitos manuscritos (MNIST).
Saida: model.h5
"""
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

EPOCAS = 5
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
CAMINHO_H5 = "model.h5"
SEED = 42


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
        Dense   (32, relu)
        Dropout (0.3)
        Dense   (10, softmax)

    Notas de projeto:
    - padding='same' preserva a dimensao espacial apos cada Conv2D e
      mantem o fluxo 28 -> 14 -> 7 exato. Evita descarte assimetrico
      de bordas.
    - Dense(32) no lugar de Dense(64). A camada densa concentra a
      maior parte dos parametros; reduzir para 32 neuronios diminui
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
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn_edge",
    )
    return modelo


def treinar(modelo, x_train, y_train):
    """Compila e treina o modelo com particao interna de validacao."""
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(
        f"[TRAIN] Treinando (epochs={EPOCAS}, batch_size={BATCH_SIZE}, "
        f"validation_split={VALIDATION_SPLIT}, optimizer=adam)..."
    )
    historico = modelo.fit(
        x_train,
        y_train,
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=2,
    )
    return historico


def imprimir_historico(historico):
    """Imprime uma tabela simples com a evolucao por epoca."""
    h = historico.history
    print("[TRAIN] Evolucao por epoca:")
    print("[TRAIN] | epoca | loss    | accuracy | val_loss | val_accuracy |")
    print("[TRAIN] |-------|---------|----------|----------|--------------|")
    for i in range(len(h["loss"])):
        print(
            f"[TRAIN] | {i + 1:^5} | "
            f"{h['loss'][i]:.4f}  | "
            f"{h['accuracy'][i]:.4f}   | "
            f"{h['val_loss'][i]:.4f}   | "
            f"{h['val_accuracy'][i]:.4f}       |"
        )


def avaliar_e_salvar(modelo, x_test, y_test):
    """Avalia o modelo no conjunto de teste e persiste o arquivo .h5."""
    _, acuracia = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"[TRAIN] Acuracia no conjunto de teste: {acuracia * 100:.2f}%")

    modelo.save(CAMINHO_H5, include_optimizer=False)
    tamanho_kb = os.path.getsize(CAMINHO_H5) / 1024
    print(f"[TRAIN] Modelo salvo: {CAMINHO_H5} ({tamanho_kb:.1f} KB)")


if __name__ == "__main__":
    keras.utils.set_random_seed(SEED)
    (x_train, y_train), (x_test, y_test) = carregar_dados()
    modelo = construir_modelo()
    modelo.summary()
    historico = treinar(modelo, x_train, y_train)
    imprimir_historico(historico)
    avaliar_e_salvar(modelo, x_test, y_test)
