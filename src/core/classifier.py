from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from glob import glob

class Builder():
    def __init__(self, img_width: int = 256, img_height: int = 256,
                 batch_size: int = 32, epochs: int = 100,
                 DATA_VALIDATION_DIR: str = None, DATA_TRAINING_DIR: str = None,
                 model:str = "CNN_model.h5", MODEL_DIR:str = None):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_classes = len(glob(DATA_TRAINING_DIR+"/*"))
        self.val_dir = DATA_VALIDATION_DIR
        self.train_dir = DATA_TRAINING_DIR
        self.model_name = model
        self.model_save_path = MODEL_DIR
    
    '''
    Esta função tem como objetivo realizar o gerador de treinos do modelo
    '''
    def train_process(self, rescale = 1/255.0,
                      rotation_range = 30,
                 
                      zoom_range = 0.4,
                      horizontal_flip = True,
                      shear_range = 0.4):
        
        train_datagen = ImageDataGenerator(rescale = rescale,
                                           rotation_range = rotation_range,
                                           zoom_range = zoom_range,
                                           horizontal_flip = horizontal_flip,
                                           shear_range = shear_range)
        
        self.train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            batch_size = self.batch_size,
                                                            class_mode = 'categorical',
                                                            target_size = (self.img_height, self.img_width))
    
    def validation_process(self, rescale = 1/255.0,
                           class_mode = 'categorical'):
        
        val_datagen = ImageDataGenerator(rescale = rescale)

        self.val_generator = val_datagen.flow_from_directory(self.val_dir,
                                                        batch_size = self.batch_size,
                                                        class_mode = class_mode,
                                                        target_size = (self.img_height, self.img_width))
    

    def model(self):
        model_name = self.model_save_path + self.model_name
        checkpoint_model = ModelCheckpoint(model_name, monitor = 'val_accuracy', verbose = 1, save_best_only = True)
        