import os
import shutil
import random

class DatasetSplitter:
    def __init__(self):
        self.paths_source = []
        self.train_dirs = {}
        self.val_dirs = {}
        self.train_set = []
        self.valid_set = []

    def create_dir(self, source):
        self.train_root_dir = os.path.abspath(os.path.join(source, "train"))
        self.val_root_dir = os.path.abspath(os.path.join(source, "validation"))

        os.makedirs(self.train_root_dir, exist_ok=True)
        os.makedirs(self.val_root_dir, exist_ok=True)

        classes = [d for d in os.listdir(source)
                   if os.path.isdir(os.path.join(source, d)) and d not in ["train", "validation"]]

        self.paths_source = []

        for class_name in classes:
            class_dir = os.path.abspath(os.path.join(source, class_name))
            self.paths_source.append(class_dir)

            train_class_dir = os.path.join(self.train_root_dir, class_name)
            val_class_dir = os.path.join(self.val_root_dir, class_name)

            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            self.train_dirs[class_name] = train_class_dir
            self.val_dirs[class_name] = val_class_dir

    def split_data(self, source, split_size=0.85):
        self.create_dir(source)
        source_parts = self.paths_source
        training = self.train_dirs
        validation = self.val_dirs

        self.train_set = []
        self.valid_set = []

        for class_dir in source_parts:
            class_name = os.path.basename(class_dir)
            all_files = [os.path.join(class_dir, f)
                         for f in os.listdir(class_dir)
                         if os.path.isfile(os.path.join(class_dir, f))]

            shuffled = random.sample(all_files, len(all_files))
            train_len = int(len(shuffled) * split_size)

            train_files = shuffled[:train_len]
            val_files = shuffled[train_len:]

            # copia arquivos de treino
            for f in train_files:
                shutil.copy(f, os.path.join(training[class_name], os.path.basename(f)))
            # copia arquivos de validação
            for f in val_files:
                shutil.copy(f, os.path.join(validation[class_name], os.path.basename(f)))

            self.train_set.extend(train_files)
            self.valid_set.extend(val_files)

        # Retorna os diretórios raiz de treino e validação
        return self.train_root_dir, self.val_root_dir

def check_dir(dir):
    # Verifica se o diretório existe e contém pelo menos uma subpasta/classe
    return os.path.isdir(dir) and len(os.listdir(dir)) > 0

# Exemplo de uso:
# dataset_dir = "CAMINHO_DO_DATASET"
# split_data = DatasetSplitter()
# train_dir, validation_dir = split_data.split_data(dataset_dir)

# if not verifica_diretorio(train_dir):
#     raise RuntimeError(f"Diretório de treino não existe ou está vazio: {train_dir}")

# if not verifica_diretorio(validation_dir):
#     raise RuntimeError(f"Diretório de validação não existe ou está vazio: {validation_dir}")
