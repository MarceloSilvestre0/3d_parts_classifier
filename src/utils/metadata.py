import os
import datetime
import json

def get_metadata (path) -> list:
    list_path = []
    # Percorre todos os arquivos na pasta
    for file_name in os.listdir(path):
        path = os.path.join(path, file_name)
        
        if os.path.isfile(path):
            # Pega metadados do arquivo
            stats = os.stat(path)

            file_size = stats.st_size
            created = datetime.datetime.fromtimestamp(stats.st_ctime)
            modified = datetime.datetime.fromtimestamp(stats.st_mtime)
            
        metadata = {
            "file_name":{file_name},
            "dt_created":{created},
            "dt_modified":{modified},
            "file_size":{file_size}
        }
        list_path.append(metadata)
    list_path = json.dumps(list_path)
    return list_path