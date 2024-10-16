import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

img = np.expand_dims(train_images[0], axis=0)

def plot_feature_maps(feature_maps):
    num_features = feature_maps.shape[-1]
    fig, axes = plt.subplots(1, num_features, figsize=(15, 15))
    
    for i in range(num_features):
        ax = axes[i]
        ax.matshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')
    
    plt.show()

model = models.Sequential(name='CNN-CIFAR10')

# Conv2D
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

_ = model(img)
# modelo intermediário apos Conv2D
model_1 = models.Model(inputs=model.inputs, outputs=model.layers[0].output)
feature_maps_1 = model_1(img)
print("Saída após Conv2D:")
plot_feature_maps(feature_maps_1)

#MaxPooling2D
model.add(layers.MaxPooling2D((2, 2)))
_ = model(img)

# modelo intermediário após MaxPooling2D
model_2 = models.Model(inputs=model.inputs, outputs=model.layers[1].output)
feature_maps_2 = model_2(img)
print("Saída após MaxPooling2D:")
plot_feature_maps(feature_maps_2)

# Conv2D
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
_ = model(img)

# modelo intermediário apos Conv2D
model_3 = models.Model(inputs=model.inputs, outputs=model.layers[2].output)
feature_maps_3 = model_3(img)
print("Saída após Conv2D:")
plot_feature_maps(feature_maps_3)

# segunda camada MaxPooling2D
model.add(layers.MaxPooling2D((2, 2)))
_ = model(img)

# modelo intermediario apos a segunda MaxPooling2D
model_4 = models.Model(inputs=model.inputs, outputs=model.layers[3].output)
feature_maps_4 = model_4(img)
print("Saída após MaxPooling2D:")
plot_feature_maps(feature_maps_4)
