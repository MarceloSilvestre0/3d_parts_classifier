import os
import open3d as o3d
import cv2

def list_file(dir):
    file_dict = {}
    for file in os.listdir(dir):
        if file.endswith(".stl"):
            file_name = os.path.splitext(file)[0]
            file_path = os.path.abspath(os.path.join(dir, file))
            file_dict[file_name] = file_path
    return file_dict

# ------------------------------------------------------------------

def render_part(stl_file, return_dir, file_name, views=12):
    mesh = o3d.io.read_triangle_mesh(stl_file)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)

    for i, angle in enumerate(range(0, 360, int(360/views))):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, angle)
        vis.poll_events()
        vis.update_renderer()

        img_path = os.path.join(return_dir, f"{file_name}_{i}.png")

        vis.capture_screen_image(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Erro ao ler {img_path}")
            continue

        img_norm = cv2.equalizeHist(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        img_clahe = clahe.apply(img_norm)

        cv2.imwrite(os.path.join(return_dir, f"img_norm_{file_name}_{i}.png"), img_norm)
        cv2.imwrite(os.path.join(return_dir, f"img_clahe_{file_name}_{i}.png"), img_clahe)

    
    vis.destroy_window()
    print(f"{views} imagens geradas para {file_name}")

#Teste
# dir = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\packing_simulator\src\data\XADREZ_REF"
# return_dir = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\packing_simulator\src\data\LFD"
# file = list_file(dir)

# print(file)

# for file_name, file_path in file.items():
#     render_part(file_path, return_dir, file_name)