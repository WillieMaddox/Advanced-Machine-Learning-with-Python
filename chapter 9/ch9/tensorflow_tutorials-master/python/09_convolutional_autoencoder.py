"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import math
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
from libs.connections import lrelu
from libs.utils import corrupt, montage
import time

def encode_weight_variable_orig(k, n_input, n_output):
    # type: (int, int, int) -> object
    """Create a weight variable."""
    initial = tf.random_uniform([k, k, n_input, n_output], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input))
    return tf.Variable(initial)


def encode_bias_variable_orig(shape):
    """Create a bias variable."""
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def decode_bias_variable_orig(shape):
    """Create a bias variable."""
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def weight_variable(shape):
    """Create a weight variable."""
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def encode_conv2d(x, W, stride):
    """2D Convolution operation."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def decode_conv2d(ci, x, W, st, shape):
    """2D Convolution operation."""
    pack = tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]])
    return tf.nn.conv2d_transpose(ci, W, pack, strides=[1, st, st, 1], padding='SAME')


def max_pool(x, dim):
    """Max pooling operation."""
    return tf.nn.max_pool(x, ksize=[1, dim, dim, 1], strides=[1, dim, dim, 1], padding='SAME')


def conv2d(x, W, stride):
    """2D Convolution operation."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool(x, dim):
    """Max pooling operation."""
    return tf.nn.max_pool(x, ksize=[1, dim, dim, 1], strides=[1, dim, dim, 1], padding='SAME')


# %%
def autoencoder(input_shape=[None, 784],
                n_filters=[1, 10, 15, 20],
                filter_sizes=[3, 5, 7],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """
    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')

    # Optionally apply denoising autoencoder
    x_noise = corrupt(x) if corruption else x

    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x_noise.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(x_noise, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x_noise.get_shape()) == 4:
        x_tensor = x_noise
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        print(layer_i, current_input.get_shape().as_list(), filter_sizes[layer_i], n_input, n_output)
        shapes.append(current_input.get_shape().as_list())
        W = encode_weight_variable_orig(filter_sizes[layer_i], n_input, n_output)
        b = encode_bias_variable_orig([n_output])
        encoder.append(W)
        output = lrelu(tf.add(encode_conv2d(current_input, W, 2), b))
        current_input = output
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(decode_conv2d(current_input, x, W, 2, shape), b))
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))
    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'encoder': encoder}






def test_mnist():

    # load MNIST as before
    mnist = input_data.read_data_sets('/media/RED6/DATA/MNIST', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder()

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Fit all training data
    batch_size = 100
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    for encoder in ae['encoder']:
        W = sess.run(encoder)
        plt.imshow(montage(W / np.max(W)), cmap='coolwarm')
        plt.waitforbuttonpress()
    plt.clf()

    # Plot example reconstructions
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(np.reshape(np.reshape(recon[example_i, ...], (784,)) + mean_img, (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
    plt.clf()


if __name__ == '__main__':
    test_mnist()
