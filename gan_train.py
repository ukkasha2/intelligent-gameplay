# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:38:31 2024

@author: UKKASHA
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm  # Import tqdm

# Load MNIST dataset
(train_images, _), (_, _) = keras.datasets.mnist.load_data()
train_images = train_images / 127.5 - 1.0  # Normalize images to the range [-1, 1]
train_images = np.expand_dims(train_images, axis=-1)

# Generator model
generator = keras.Sequential([
    layers.Dense(7 * 7 * 256, input_shape=(100,), use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Reshape((7, 7, 256)),

    layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])

# Discriminator model
discriminator = keras.Sequential([
    layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
    layers.LeakyReLU(),
    layers.Dropout(0.3),

    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.LeakyReLU(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(1)
])

# Compile discriminator
discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss=keras.losses.BinaryCrossentropy(from_logits=True))

# Combine generator and discriminator into a GAN model
discriminator.trainable = False  # Freeze discriminator during GAN training
gan = keras.Sequential([generator, discriminator])
gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss=keras.losses.BinaryCrossentropy(from_logits=True))

# Training loop
epochs = 10
batch_size = 64
noise_dim = 100
steps_per_epoch = train_images.shape[0] // batch_size

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for step in tqdm(range(steps_per_epoch)):  # tqdm added here
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generated_images = generator.predict(noise)

        real_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_labels)

    # Print epoch progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")

        generated_images = (generator.predict(np.random.normal(0, 1, (16, noise_dim))) + 1) / 2.0
        fig, axs = plt.subplots(4, 4)
        count = 0
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.show()

generator.save("/content/gdrive/MyDrive/generator_model.h5")
print('model saved')


from tensorflow.keras.models import load_model

# Load the generator model
generator = load_model("/content/gdrive/MyDrive/generator_model.h5", compile=False)

# Compile the loaded model manually
generator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))

# Save the compiled model again
generator.save("/content/gdrive/MyDrive/generator_model_2.h5")
print('model saved')