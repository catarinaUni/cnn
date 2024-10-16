import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]], fontsize=15)
plt.show()

def create_model():
    model = models.Sequential(name='CNN-CIFAR10')
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True)

fold_accuracies = []
fold_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_images, train_labels)):
    print(f"Treinando no Fold {fold+1}/{num_folds}")

    X_train_fold, X_val_fold = train_images[train_idx], train_images[val_idx]
    y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]

    model = create_model()

    history = model.fit(X_train_fold, y_train_fold, epochs=5, 
                        validation_data=(X_val_fold, y_val_fold), verbose=1)

    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_losses.append(val_loss)
    fold_accuracies.append(val_accuracy)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.ylim([0.5, 1])
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(loc='upper right')

    plt.suptitle(f"Fold {fold + 1}")
    plt.show()

print("\nResultados da validação cruzada:")
print(f"Média da acurácia de validação: {np.mean(fold_accuracies):.4f}")
print(f"Média da perda de validação: {np.mean(fold_losses):.4f}")
