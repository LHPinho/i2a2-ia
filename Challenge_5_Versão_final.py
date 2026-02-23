import os
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import copy2

# Define the model
def define_model():
    # Load VGG16 model without the top classification layers
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # Mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # Add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # Define new model
    model = Model(inputs=model.inputs, outputs=output)
    # Compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train the model
def run_test_harness():
    # Define model
    model = define_model()
    # Create data generators
    datagen = ImageDataGenerator(featurewise_center=True)
    # Specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # Prepare iterators
    train_it = datagen.flow_from_directory('C:/Users/Colaborador/OneDrive - SMARTHIS CONSULTORIA EM TECNOLOGIA DA INFORMAÇÃO LTDA/Área de Trabalho/Challenge 5 I2A2/dataset/dataset/training_set',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))
    test_it = datagen.flow_from_directory('C:/Users/Colaborador/OneDrive - SMARTHIS CONSULTORIA EM TECNOLOGIA DA INFORMAÇÃO LTDA/Área de Trabalho/Challenge 5 I2A2/dataset/dataset/test_set',
                                          class_mode='binary', batch_size=64, target_size=(224, 224), shuffle=False)
    # Fit model
    model.fit(train_it, steps_per_epoch=len(train_it), epochs=1, verbose=1)
    # Save model
    model.save('final_model.h5')
    return model, test_it

# Evaluate the model and calculate the confusion matrix
def evaluate_model(model, test_it):
    # Predict the classes for the test set
    predictions = model.predict(test_it, steps=len(test_it), verbose=1)
    predicted_classes = (predictions > 0.5).astype(int)
    true_classes = test_it.classes
    class_labels = list(test_it.class_indices.keys())

    # Calculate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print('Confusion Matrix')
    print(cm)
    print('Classification Report')
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Run the training and evaluation
if __name__ == "__main__":
    model, test_it = run_test_harness()
    evaluate_model(model, test_it)