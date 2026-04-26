# Relatório do Candidato

**Nome completo:** Thiago Nerton Macedo Alves  
**GitHub:** https://github.com/Nertonm/processoseletivoIA

## 1. Resumo da arquitetura do modelo

O modelo implementado em `train_model.py` é uma CNN pequena para classificação
dos dígitos do MNIST. Como as imagens têm só 28x28 pixels e um canal de cinza,
optei por uma arquitetura direta: duas convoluções, duas reduções espaciais com
pooling, uma camada densa compacta e a saída com 10 classes.

| Etapa | Camada | Configuração | Função |
|-------|--------|--------------|--------|
| Entrada | `Input` | `(28, 28, 1)` | Recebe a imagem normalizada com o canal explícito |
| 1 | `Conv2D` | 32 filtros, kernel 3x3, ReLU, `padding="same"` | Extrai traços simples, como bordas e curvas |
| 2 | `MaxPooling2D` | pool 2x2 | Reduz de 28x28 para 14x14 |
| 3 | `Conv2D` | 64 filtros, kernel 3x3, ReLU, `padding="same"` | Combina os traços iniciais em formas mais completas |
| 4 | `MaxPooling2D` | pool 2x2 | Reduz de 14x14 para 7x7 |
| 5 | `Flatten` | 7x7x64 | Transforma os mapas de características em vetor |
| 6 | `Dense` | 32 unidades, ReLU | Faz a combinação final com menos parâmetros |
| 7 | `Dropout` | taxa 0.3 | Reduz overfitting durante o treino |
| Saída | `Dense` | 10 unidades, softmax | Retorna a probabilidade de cada dígito |

Usei `padding="same"` nas convoluções para manter o tamanho espacial antes do
pooling. Isso deixa o caminho do tensor bem previsível: 28 -> 14 -> 7, sem
perder bordas de forma assimétrica.

A sequência `Conv2D(32) -> Conv2D(64)` é um pouco maior do que uma CNN mínima,
mas ajuda a separar dígitos visualmente parecidos. A primeira convolução aprende
traços básicos; a segunda trabalha combinações desses traços. O `Dropout(0.3)`
fica na parte densa, onde há mais risco de overfitting, e não tem custo na
inferência.

Também testei a camada densa com 64 unidades. Ela deu uma acurácia ligeiramente
maior, mas quase dobrou o número de parâmetros. Como o foco do desafio é Edge AI,
mantive `Dense(32)`: o modelo ficou bem menor e a perda de acurácia foi mínima.

O treino usa `Adam`, `sparse_categorical_crossentropy`, métrica `accuracy`, 5
épocas, `batch_size=64` e `validation_split=0.1`. Fixei
`keras.utils.set_random_seed(42)` para facilitar a reprodução dos resultados. O
modelo é salvo como `model.h5` com `include_optimizer=False`, já que o estado do
otimizador não é necessário para inferência.

## 2. Bibliotecas utilizadas

| Biblioteca | Versão declarada | Uso no projeto |
|------------|------------------|----------------|
| TensorFlow / Keras | `>=2.12,<2.16` | MNIST, definição da CNN, treino, salvamento `.h5`, conversão TFLite e validação com `tf.lite.Interpreter` |
| NumPy | `<2.0` | Normalização das imagens, ajuste do canal de entrada, `argmax` e cálculo de acurácia |
| `os` | biblioteca padrão | Leitura do tamanho dos arquivos gerados |

Limitei o TensorFlow à faixa `>=2.12,<2.16` para evitar mudanças da transição
para Keras 3. O `numpy<2.0` acompanha essa escolha de compatibilidade.

## 3. Técnica de otimização do modelo

No `optimize_model.py`, a conversão para TensorFlow Lite usa **Dynamic Range
Quantization**:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
modelo_tflite = converter.convert()
```

Essa técnica reduz os pesos do modelo no arquivo final, normalmente para `int8`,
e mantém as ativações tratadas de forma dinâmica durante a inferência. Escolhi
essa opção porque ela reduz bastante o tamanho do arquivo sem exigir um
`representative_dataset`, o que deixa o fluxo mais simples e fácil de rodar no
ambiente de avaliação.

Depois da conversão, o `.tflite` é validado nas 10.000 imagens do conjunto de
teste do MNIST. O script roda o modelo com `tf.lite.Interpreter`, confere o
formato da primeira saída `(1, 10)`, verifica se não há `NaN` ou `Inf` e compara
a acurácia do TFLite com a do modelo Keras no mesmo conjunto. Assim, a conversão
não é tratada só como geração de arquivo; ela também é checada na prática.

## 4. Resultados obtidos

Com a arquitetura final (`Dense(32)`), o modelo ficou pequeno e manteve boa
acurácia para MNIST:

| Indicador | Resultado |
|-----------|-----------|
| Arquitetura | 2 `Conv2D`, 2 `MaxPooling2D`, `Dense(32)`, `Dropout(0.3)` e saída `softmax` |
| Parâmetros treináveis | 119.530 |
| Acurácia Keras no teste completo | 98,95% |
| Acurácia TFLite no teste completo | 98,98% |
| Delta Keras -> TFLite | +0,03 ponto percentual |
| Tamanho do `model.h5` | 489,9 KB |
| Tamanho do `model.tflite` | 123,2 KB |
| Redução de tamanho | 74,9% |

Antes de fechar a arquitetura, comparei `Dense(64)` com `Dense(32)` usando a
mesma seed, o mesmo número de épocas e a mesma validação no teste completo:

| Versão | Parâmetros | Keras | TFLite | `model.h5` | `model.tflite` |
|--------|------------|-------|--------|------------|----------------|
| `Dense(64)` | 220.234 | 98,99% | 99,00% | 882,4 KB | 222,6 KB |
| `Dense(32)` | 119.530 | 98,95% | 98,98% | 489,9 KB | 123,2 KB |

A troca para `Dense(32)` reduziu cerca de 45,7% dos parâmetros, 44,5% do `.h5`
e 44,7% do `.tflite`. A perda foi de 0,04 ponto percentual no Keras e 0,02 ponto
percentual no TFLite. Para um modelo voltado a Edge AI, considerei essa troca
mais interessante do que manter a versão maior só por uma diferença muito pequena
de acurácia.

## 5. Comentários adicionais

O principal gargalo de tamanho estava na transição `Flatten -> Dense`. Depois do
segundo pooling, o tensor tem `7x7x64 = 3136` valores. Ligar esse vetor a 64
neurônios cria muitos pesos; com 32 neurônios, essa parte cai quase pela metade
e ainda sobra capacidade para o MNIST.

O trade-off é simples: uma camada densa maior pode capturar combinações mais
ricas, mas aumenta arquivo, memória e custo de inferência. Como o conjunto MNIST
é relativamente simples, a versão menor preservou a acurácia quase no mesmo nível
e ficou mais adequada ao objetivo de um modelo embarcado.

**Possíveis caminhos conforme o hardware alvo**

| Hardware alvo | Caminho possível | Vantagem | Custo ou risco |
|---------------|------------------|----------|----------------|
| CPU comum ou Raspberry Pi | Manter Dynamic Range Quantization | Simples, compatível e já reduz bem o arquivo | Ganho de latência pode ser limitado |
| Microcontrolador / TFLite Micro | Testar Full Integer Quantization com `representative_dataset` | Melhor uso de RAM/flash e caminho mais próximo de `int8` real | Exige calibração e nova validação |
| EdgeTPU ou acelerador `int8` | Full Integer Quantization e checagem das operações suportadas | Maior chance de aceleração em hardware dedicado | Nem toda operação TFLite é aceita pelo compilador do acelerador |
| GPU/DSP com suporte a FP16 | Testar Float16 Quantization | Pode reduzir tamanho e melhorar execução nesse tipo de hardware | Reduz menos que `int8` e depende do delegate |
| Foco máximo em acurácia | Voltar para `Dense(64)`, treinar mais épocas ou ajustar learning rate | Pode recuperar pequenos ganhos | Aumenta tamanho e custo de inferência |
| Foco máximo em tamanho | Testar `Dense(16)`, `GlobalAveragePooling2D`, `SeparableConv2D` ou pruning | Pode deixar o modelo bem menor | Maior risco de perder acurácia |

Para esta entrega, mantive `Dense(32)` com Dynamic Range Quantization porque é
uma solução simples de reproduzir, pequena o suficiente para o escopo do desafio
e validada no conjunto completo de teste.
