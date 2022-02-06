import warnings
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize model using DenseNet121 (121 layers)
#
model_d = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Add layers to model
x = model_d.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

#
preds = Dense(10, activation='softmax')(x)

# Create
model = Model(inputs=model_d.input, outputs=preds)
# model.summary()

for layer in model.layers[:-8]:
    layer.trainable = False

for layer in model.layers[-8:]:
    layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

feature_data = np.load('feature_data.npy')
print(feature_data.shape)
feature_data_10 = feature_data[:2498]

labels = np.load('labels.npy')
labels_10 = labels[:2498]

label_bin = LabelBinarizer()
labels_10 = label_bin.fit_transform(labels_10)

x_train, x_test, y_train, y_test = train_test_split(feature_data_10, labels_10, test_size=0.3, random_state=44,
                                                    stratify=labels_10)

x_train_train, x_val, y_train_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=44,
                                                              stratify=y_train)

anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)

checkpoint = ModelCheckpoint('model3.h5', verbose=1, save_best_only=True)

datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True, shear_range=0.2)

datagen.fit(x_train)
# Fits-the-model
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                              steps_per_epoch=x_train.shape[0] // 128,
                              epochs=100,
                              verbose=2,
                              callbacks=[anne, checkpoint],
                              validation_data=(x_test, y_test))
