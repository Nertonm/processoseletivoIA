"""
Conversao do modelo treinado para TFLite com Dynamic Range Quantization.
Entrada: model.h5
Saida: model.tflite
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

CAMINHO_H5 = "model.h5"
CAMINHO_TFLITE = "model.tflite"
AMOSTRAS_VALIDACAO = 100


def carregar_modelo(caminho):
    """Carrega o modelo Keras treinado do disco."""
    print(f"[OPTIMIZE] Carregando {caminho}...")
    return keras.models.load_model(caminho)


def converter_para_tflite(modelo, caminho_saida):
    """Converte o modelo Keras para TFLite com dynamic range quantization."""
    print("[OPTIMIZE] Aplicando Dynamic Range Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    modelo_tflite = converter.convert()

    with open(caminho_saida, "wb") as f:
        f.write(modelo_tflite)
    print(f"[OPTIMIZE] Modelo salvo: {caminho_saida}")


def validar_inferencia(caminho_tflite, modelo_keras=None):
    """Roda o tflite em AMOSTRAS_VALIDACAO imagens do MNIST e compara com o Keras."""
    print("[OPTIMIZE] Validando inferencia TFLite...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    imagens = np.expand_dims(
        (x_test[:AMOSTRAS_VALIDACAO] / 255.0).astype(np.float32), axis=-1
    )
    rotulos = y_test[:AMOSTRAS_VALIDACAO]

    interpreter = tf.lite.Interpreter(model_path=caminho_tflite)
    interpreter.allocate_tensors()
    indice_entrada = interpreter.get_input_details()[0]["index"]
    indice_saida = interpreter.get_output_details()[0]["index"]

    predicoes_tflite = np.zeros(AMOSTRAS_VALIDACAO, dtype=np.int64)
    for i in range(AMOSTRAS_VALIDACAO):
        interpreter.set_tensor(indice_entrada, imagens[i : i + 1])
        interpreter.invoke()
        saida = interpreter.get_tensor(indice_saida)
        if i == 0:
            assert saida.shape == (1, 10), f"shape inesperado: {saida.shape}"
            assert np.isfinite(saida).all(), "saida tem NaN ou Inf"
        predicoes_tflite[i] = np.argmax(saida)

    acuracia_tflite = (predicoes_tflite == rotulos).mean() * 100
    print(
        f"[OPTIMIZE] Acuracia TFLite ({AMOSTRAS_VALIDACAO} amostras): "
        f"{acuracia_tflite:.2f}%"
    )

    if modelo_keras is not None:
        predicoes_keras = np.argmax(
            modelo_keras.predict(imagens, verbose=0), axis=1
        )
        acuracia_keras = (predicoes_keras == rotulos).mean() * 100
        print(
            f"[OPTIMIZE] Acuracia Keras ({AMOSTRAS_VALIDACAO} amostras): "
            f"{acuracia_keras:.2f}%"
        )
        print(f"[OPTIMIZE] Delta: {acuracia_tflite - acuracia_keras:+.2f} pp")
