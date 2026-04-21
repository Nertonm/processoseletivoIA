"""
Conversao do modelo treinado para TFLite com Dynamic Range Quantization.
Entrada: model.h5
Saida: model.tflite
"""
import tensorflow as tf
from tensorflow import keras

CAMINHO_H5 = "model.h5"
CAMINHO_TFLITE = "model.tflite"


def carregar_modelo(caminho):
    """Carrega o modelo Keras treinado do disco."""
    print(f"[OPTIMIZE] Carregando {caminho}...")
    return keras.models.load_model(caminho)


def converter_para_tflite(modelo, caminho_saida):
    """Converte o modelo Keras para TFLite com Dynamic Range Quantization.

    A tecnica reduz os pesos de float32 para int8 durante a conversao,
    sem dados de calibracao. As ativacoes permanecem em float e sao
    quantizadas dinamicamente em tempo de inferencia.
    """
    print("[OPTIMIZE] Aplicando Dynamic Range Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    modelo_tflite = converter.convert()

    with open(caminho_saida, "wb") as f:
        f.write(modelo_tflite)
    print(f"[OPTIMIZE] Modelo salvo: {caminho_saida}")
