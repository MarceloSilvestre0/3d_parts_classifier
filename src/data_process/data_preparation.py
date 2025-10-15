import pandas as pd

list_bounding = [{'NM_PART': 'bishop_ref.stl', 'WIDTH': 40.0, 'HEIGHT': 40.0, 'LENGTH': 76.0},{'NM_PART': 'bishop_ref.stl', 'WIDTH': 40.0, 'HEIGHT': 40.0, 'LENGTH': 76.0}]
list_classes = ['Bishop', 'King', 'Knight', 'Pawn', 'Bishop', 'Rook']

colunas = ["NM_PART",
           "WIDTH", "HEIGHT", "LENGTH"]

list_bounding_data = pd.DataFrame(list_bounding, columns=colunas)
print(list_bounding_data)