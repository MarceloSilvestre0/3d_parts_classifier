''''
Melhorias futuras:
1. Criar uma função para coletar os dados de saída da classificação
2. Criar uma função para subir os dados em um banco
'''


import os
from dotenv import load_dotenv
from core.MathPartProcess import MathPartProcess
from core.classifier import STLPartClassifier
import pandas as pd

#Carrega as varivaies de ambiente
load_dotenv()

#Realiza a extração das imagens do arquivo 3D - LFD
output_directory = os.getenv("OUTPUT_DIRECTORY")
input_directory = os.getenv("INPUT_DIRECTORY")

#Inicializa a classe
lfd_catcher = MathPartProcess()

#Inicializa o classificador
classifier = STLPartClassifier()

#Executa o método
#A bounding box irá iterar no diretório com as peças .STL e irá extrair os dados do modelo 3D
#A saida do Bounding é uma lista de dicionários com os dados de cada peça
bounding = lfd_catcher.execute_render(input_directory, output_directory, views=2)

#Lista os itens presentes no diretório
lfd = os.listdir(output_directory)

#Inicializa a lista de classificações
list_classification = []

#Laço para iterar nas imagens geradas pelo lfd
for part in range(len(lfd)):
    #Monta os caminhos de cada uma das peças
    path_part = os.path.join(output_directory, lfd[part])

    #Executa a predição de cada uma das peças
    classes = classifier.model_predict(path_part)
    list_classification.append(classes)

print(bounding)
print(list_classification)



