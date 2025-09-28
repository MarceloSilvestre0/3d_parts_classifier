import open3d as o3d
import numpy as np

def bouding_box(file_path) -> dict:
    # Origem Open3d -> Carrega a malha
    mesh = o3d.io.read_triangle_mesh(file_path)

    # Origem Open3d -> Bounding box
    bounding_box = mesh.get_axis_aligned_bounding_box()

    # Origem Open3d -> Calcula as dimensões
    min_bound = bounding_box.get_min_bound()
    max_bound = bounding_box.get_max_bound()
    dimensoes = np.array(max_bound) - np.array(min_bound)

    #Retorna um dicionário com as medidas
    data = {
        "Largura (X)": round(float(dimensoes[0]), 2),
        "Altura (Y)": round(float(dimensoes[1]),2),
        "Comprimento (Z)": round(float(dimensoes[2]),2)
    }
    return data

file_path = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\packing_simulator\src\data\XADREZ_REF\bishop_ref.stl"
#C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\packing_simulator\src\data\XADREZ_REF\bishop_ref.stl
lista = bouding_box(file_path)

print(lista)