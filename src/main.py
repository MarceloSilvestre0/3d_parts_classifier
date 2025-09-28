# from src.utils.data_ingestion import create_table
# from src.utils.metadata import get_metadata
# from src.utils.data_ingestion import save_data
# from src.utils.database import Base, chess

from core.MathPartProcess import MathPartProcess

processador = MathPartProcess()

output_directory = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\3d_parts_classifier\teste"
input_directory = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\XADREZ_REF"
views = 2

print("rodando")
processador.execute_render(input_directory, output_directory, views)