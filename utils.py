
import tensorflow as tf
import numpy as np
import cv2

def softmax(x):
    exp_x = np.exp(x) 
    return exp_x / exp_x.sum(axis=0)

def calculate_grad_cam(model, layer_name, image):
    # Create a graph that outputs target convolution and output
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])

    # Get the score for target class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[0]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    # Heatmap visualization
    cam = cv2.resize(cam.numpy(), (180, 240))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    return cam

