import pandas as pd

def data_export(data):
    df = pd.DataFrame([data])
    df.to_csv("saida2.csv",index=False, encoding="utf-8-sig")
    print(f"Arquivo gerado com sucesso!")