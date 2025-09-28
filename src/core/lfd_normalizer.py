import cv2
import glob
import os

pasta_dataset = r"C:\Users\Marcelo Silvestre\Documents\1.Estudos\1.Pos_Graduacao\TCC\TCC\packing_simulator\src\data\LFD"
tamanho = (224,224)
for pasta in os.listdir(pasta_dataset):
    arquivos = glob.glob(os.path.join(pasta_dataset, pasta, "*.png"))
    for arq in arquivos:
        img = cv2.imread(arq)
        img = cv2.resize(img, tamanho)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(arq, img)
