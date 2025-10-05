import open3d as o3d
import numpy as np
import cv2
import os

class MathPartProcess:
    def __init__(self):
        self.mesh = None
        self.data = {}
    
    def execute_render(self, directory:str, return_dir:str, views):
        #Extrai e retorna o dicionário {file_name: file_path}
        files = self.__list_file(directory)
        print(directory)

        #Captura o dicionário e itera a função 'render_part'.
        for file_name, file_path in files.items():
            self.render_part(file_path, return_dir, file_name, views=views)
            self.calculate(file_path)

    '''
    Esta função irá retornar o dicionário com os caminhos das peças dentro do diretório
    '''

    def __list_file(self, directory:str) -> dict:
        file_dict = {}
        for file in os.listdir(directory):
            if file.endswith(".stl"):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.abspath(os.path.join(directory, file))
                file_dict[file_name] = file_path
        return file_dict
        
    '''
    Esta função irá ler os caminhos listados na função anterior e salvará as fotos extraídas no diretório de avaliação
    '''

    def render_part(self, stl_file, return_dir, file_name, views):

        try:
            self.mesh = o3d.io.read_triangle_mesh(stl_file)
            self.mesh.compute_vertex_normals()
            
        except(IsADirectoryError, FileNotFoundError) as e:
            print(f"Caminho inválido ou arquivo inexistente: {e}")
            return None
        
        except Exception as e:
            print(f"Erro ao ler o arquivo: {e}")

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(self.mesh)

        for i, angle in enumerate(range(270, 360, int(360/views))):
            ctr = vis.get_view_control()
            ctr.rotate(0.0, -angle)
            vis.poll_events()
            vis.update_renderer()

            img_path = os.path.join(return_dir, f"{file_name}_{i}.png")

            vis.capture_screen_image(img_path)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Erro ao ler {img_path}")
                continue

            # img_norm = cv2.equalizeHist(img)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            # img_clahe = clahe.apply(img_norm)

            # # cv2.imwrite(os.path.join(return_dir, f"img_norm_{file_name}_{i}.png"), img_norm)
            # cv2.imwrite(os.path.join(return_dir, f"img_clahe_{file_name}_{i}.png"), img_clahe)

        vis.destroy_window()

    
    '''Função para calcular o bounding box do arquivo 3D
    Este método irá receber o caminho o arquivo 3D (.stl), irá processar suas malhas e retornar os valores,
    em mm dos eixos delimitadores do produto. Dessa forma, é possivel calcular o volume do ocupação da peça    
    '''
    def calculate(self, file_path:str) -> dict:
        try:
            self.mesh = o3d.io.read_triangle_mesh(file_path)
        except(IsADirectoryError, FileNotFoundError) as e:
            print(f"Caminho inválido ou arquivo inexistente: {e}")
            return None
        
        except Exception as e:
            print(f"Erro ao ler o arquivo: {e}")

        bounding_box = self.mesh.get_axis_aligned_bounding_box()
        min_bound = bounding_box.get_min_bound()
        max_bound = bounding_box.get_max_bound()
        dimensions = np.array(max_bound) - np.array(min_bound)

        self.data = {
            "Largura (X)": round(float(dimensions[0]), 2),
            "Altura (Y)": round(float(dimensions[1]),2),
            "Comprimento (Z)": round(float(dimensions[2]),2)
        }
        return self.data
