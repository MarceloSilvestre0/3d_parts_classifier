'''
Validado!
Esta classe e seus métodos estão funcionando.
Ao chamar o método split_data, dentro de DatasetSplitter, é realizado a criação dos diretórios e a separaçao de dados
de teste e validação
'''
from data_split import DatasetSplitter

splitter = DatasetSplitter()
caminho = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\3d_parts_classifier\src\data\dataset"
splitter.split_data(caminho)