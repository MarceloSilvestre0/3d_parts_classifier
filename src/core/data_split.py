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
        
        for filename in os.listdir(source):
            file = os.path.abspath(os.path.join(source, filename))

            if os.path.isdir(file):  # só processa diretórios (classes)
                self.paths_source.append(file)

                # cria pastas de treino e validação para essa classe
                train_class_dir = os.path.abspath(os.path.join(source, "train", filename))
                val_class_dir = os.path.abspath(os.path.join(source, "validation", filename))

                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(val_class_dir, exist_ok=True)

                self.train_dirs[filename] = train_class_dir
                self.val_dirs[filename] = val_class_dir

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

        return self.train_set, self.valid_set


