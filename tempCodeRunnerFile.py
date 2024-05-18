import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
model = load_model('model.h5')
# Compile model with a lower learning rate
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Define a function to generate Grad-CAM heatmap
def generate_grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    output = conv_output[0]
    grads = tape.gradient(loss, conv_output)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = np.maximum(cam, 0)
    cam /= np.max(cam)

    return cam

shape = (1000, 19)

# Generate a random array of the specified shape
sample_image= np.random.rand(*shape)

# Print layer names
for layer in model.layers:
  print(layer.name)

# Generate Grad-CAM heatmap
# Generate Grad-CAM heatmap using the first convolutional layer
cam = generate_grad_cam(model, np.expand_dims(sample_image, axis=0), 'conv1d_8')
# Resize the heatmap to match the dimensions of the input image
heatmap = tf.image.resize(np.expand_dims(cam, axis=-1), (sample_image.shape[0], 1))

# Transpose the heatmap to rotate it by 90 degrees
heatmap_transposed = np.transpose(heatmap[:, :, 0])

print("Heatmap shape:", heatmap_transposed.shape)
print("Sample image shape:", sample_image.shape)


# Plot the original EEG signals and Grad-CAM heatmap
plt.figure(figsize=(14, 10))

# Create subplot for Grad-CAM heatmap with EEG signals overlay
plt.imshow(heatmap_transposed, cmap='jet', alpha=0.6, aspect='auto', extent=[0, sample_image.shape[0], -5, 5])
plt.plot(sample_image, color='black', alpha=0.5)  # Overlay the original EEG signals on the heatmap
plt.title('Grad-CAM with EEG Signals Overlay')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.colorbar(label='Importance')
plt.grid(True)

plt.show()

