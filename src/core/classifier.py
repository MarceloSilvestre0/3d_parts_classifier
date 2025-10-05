from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class STLPartClassifier():
    def __init__(self, img_width: int = 256, img_height: int = 256,
                 batch_size: int = 32, epochs: int = 100,
                 validation_dir: str = "", training_dir: str = "",
                 model:str = "CNN_model.h5", MODEL_DIR:str = "", verbose: int = 1):
        
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_classes = len(glob(training_dir+"/*"))
        self.val_dir = validation_dir
        self.train_dir = training_dir
        self.model_name = model
        self.model_save_path = MODEL_DIR
        self.verbose = verbose

    '''
    Esta função tem como objetivo realizar o gerador de treinos do modelo
    '''
    def train_process(self, source, rescale=1/255.0,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        brightness_range=[0.8, 1.2],
                        fill_mode='nearest'):
        
        source = self.train_dir
        train_datagen = ImageDataGenerator(rescale=rescale,
                                            rotation_range=rotation_range,
                                            width_shift_range=width_shift_range,
                                            height_shift_range=height_shift_range,
                                            shear_range=shear_range,
                                            zoom_range=zoom_range,
                                            horizontal_flip=horizontal_flip,
                                            brightness_range=brightness_range,
                                            fill_mode=fill_mode)
        
        self.train_generator = train_datagen.flow_from_directory(source,
                                                            batch_size = self.batch_size,
                                                            class_mode = 'categorical',
                                                            target_size = (self.img_height, self.img_width))
    
    def validation_process(self, rescale = 1/255.0,
                           class_mode = 'categorical'):
        source = self.val_dir
        
        val_datagen = ImageDataGenerator(rescale = rescale)

        self.val_generator = val_datagen.flow_from_directory(source,
                                                        batch_size = self.batch_size,
                                                        class_mode = class_mode,
                                                        target_size = (self.img_height, self.img_width))
    

    def model_architecture(self):
        self.callback = EarlyStopping(monitor='val_loss', patience=100, verbose=self.verbose, restore_best_weights=True)
        self.model_path = self.model_save_path + self.model_name
        self.checkpoint_model = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=self.verbose, save_best_only=True)

        # Carrega o backbone pré-treinado
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))
        base_model.trainable = False  # Congela as camadas do backbone

        # Adiciona camadas customizadas
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model, self.checkpoint_model
   
    # def model_architecture(self):

    #     self.callback = EarlyStopping(monitor='val_loss', patience=100, verbose=self.verbose, restore_best_weights=True)
    #     self.model_path = self.model_save_path + self.model_name
    #     self.checkpoint_model = ModelCheckpoint(self.model_path, monitor = 'val_accuracy', verbose = self.verbose, save_best_only = True)

    #     self.model = Sequential([
    #         Conv2D(32, (3,3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
    #         MaxPooling2D(2,2),
            


    #         Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    #         MaxPooling2D(2,2),
            

    #         Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    #         MaxPooling2D(2,2),

    #         Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    #         MaxPooling2D(2,2),
           
    #         Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    #         MaxPooling2D(2,2),
    #         BatchNormalization(),

    #         Conv2D(256, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    #         MaxPooling2D(2,2),
    #         BatchNormalization(),

    #         Flatten(),

    #         Dense(512, activation='relu'),

    #         Dropout(0.2),

    #         Dense(self.num_classes, activation='softmax')
    #         ])
        
    #     self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #     return self.model, self.checkpoint_model
    
    def model_fit(self):
        self.model_fit = self.model.fit(self.train_generator,
                                    epochs = self.num_epochs,
                                    verbose = self.verbose,
                                    validation_data = self.val_generator,
                                    callbacks = [self.checkpoint_model, self.callback])
        return self.model_fit
    
    def prepare_image(self, image_path):
        image = load_img(image_path, target_size=(self.img_height, self.img_width))
        image_result = img_to_array(image)
        self.image_result = np.expand_dims(image_result, axis=0)
        return self.image_result
    
    def model_predict(self, png_path: str = "", classes: list = []):
        model = load_model(self.model_path) #inserir no execute
        image = png_path
        image_prepare = self.prepare_image(image)

        image_array = model.predict(image_prepare,
                              batch_size = self.batch_size,
                              verbose = self.verbose)
        output = np.argmax(image_array, axis = 1)

        self.classification = classes[output[0]]

        return self.classification
    
    def auditory(self, X_val=None, y_val=None):
        history = self.model_fit
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        # Gráfico de acurácia
        plt.figure(figsize=(14,7))
        plt.plot(epochs, acc, 'r-', label="Train Accuracy")
        plt.plot(epochs, val_acc, 'b-', label="Validation Accuracy")
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.title('Acurácia de treino e validação')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        # Gráfico de perda
        plt.figure(figsize=(14,7))
        plt.plot(epochs, loss, 'r-', label="Train Loss")
        plt.plot(epochs, val_loss, 'b-', label="Validation Loss")
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.title('Perda de treino e validação')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # Impressão das estatísticas finais
        print(f"Acurácia final (treino): {acc[-1]:.4f}")
        print(f"Acurácia final (validação): {val_acc[-1]:.4f}")
        print(f"Perda final (treino): {loss[-1]:.4f}")
        print(f"Perda final (validação): {val_loss[-1]:.4f}")

        # Estatísticas detalhadas (se dados de validação disponíveis)
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = y_val.argmax(axis=1) if len(y_val.shape) > 1 else y_val

            print(classification_report(y_true, y_pred_classes))
            cm = confusion_matrix(y_true, y_pred_classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.show()
    


