# from src.utils.data_ingestion import create_table
# from src.utils.metadata import get_metadata
# from src.utils.data_ingestion import save_data
# from src.utils.database import Base, chess
from core.data_split import DatasetSplitter
import os
from dotenv import load_dotenv
from core.MathPartProcess import MathPartProcess
from core.classifier import STLPartClassifier
from core.data_split import check_dir
#Carrega o Dotenv
load_dotenv()

#Realiza a extração das imagens do arquivo 3D - LFD
output_directory = os.getenv("OUTPUT_DIRECTORY")
input_directory = os.getenv("INPUT_DIRECTORY")

#Inicializa a classe
lfd_catcher = MathPartProcess()

#Executa o método
lfd_catcher.execute_render(input_directory, output_directory, views=2)

#Realiza o Split Data do Dataset
dataset_dir = os.getenv("DATASET_DIR")
split_data = DatasetSplitter()

#Inicializa as variaveis train_dir e validation_dir com os caminhos de treino e validação
train_dir, validation_dir = split_data.split_data(dataset_dir)

if not check_dir(train_dir):
    raise RuntimeError(f"Diretório de treino não existe ou está vazio: {train_dir}")

if not check_dir(validation_dir):
    raise RuntimeError(f"Diretório de validação não existe ou está vazio: {validation_dir}")

print(f"Train dir: {train_dir}, Validation dir: {validation_dir}")

#Realiza o treino do modelo e salva o modelo.h5
classifier = STLPartClassifier(training_dir=train_dir, validation_dir=validation_dir)
classifier.train_process(train_dir)
classifier.validation_process()
classifier.model_architecture()
classifier.model_fit()

#Realiza a classificação das peças geradas


'''Antes de tudo abaixo:
    1. É necessário revisar a classe data_split, pois a mesma precisa ser ajustada para ler o dataset através das variáveis globais
    em seguida deve criar e separar os arquivos de imagens nos diretórios dentro de src/data/data_process

    2. Após isso, garanta que estes diretórios estarão configurados como variáveis globais dentro da classe classificadora

    3. Garanta também que a pasta destino dos arquivos lidos pela classe 'MathPartProcess' estão sendo salvos em um diretório isolado
    e com varaiável global definida dentro do .env

Depois que ajustar as etapas anteriores, pode seguir essas etapas, pois serão necessárias
A partir daqui o algoritmo deve iterar em "OUTPUT_DIRECTORY" e realizar o fit() e depois o predict() de cada um dos arquivos

Após a clissificação, o algoritmo deve extrair os valores de bouding box de cada um dor produtos]
Após a etapa anterior o algoritmo deve extrair os metadados dos arquivos
Após o anterior o algoritmo deve tratar esses dados e organizalos em um dataframe
Após realizar a etapa anterior o algoritmo deve subir esses dados em um banco PostgreSQL
'''
