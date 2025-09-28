from src.utils.data_ingestion import create_table
from src.utils.metadata import get_metadata
from src.utils.data_ingestion import save_data
from src.utils.database import Base, chess


if __name__ == "__main__":
    create_table()
    print("Iniciando a criação de um novo registro")
    try:
        path = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\packing_simulator\src\data\XADREZ_REF"
        data = get_metadata(path)
        if data:
            save_data(data)
    except KeyboardInterrupt:
        print("\nProcesso interrompido pelo usuário")
    except Exception as e:
        print(f"Erro durante a execução: {e}")