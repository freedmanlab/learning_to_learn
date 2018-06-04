import numpy as np
import tensorflow as tf
from parameters import par
import pickle
import task
import os


# TODO: move these parameters to a better home
num_layers = len(par['conv_filters'])
dense_layers = [2048, 1000, 100] # 2048 is the size after the convolutional layers
training_iterations = 20000
dense_layers_spatial = [16384, 24] #16384 = 32*32*16 (product of dimensions of output of first conv layer)

use_gpu = True
if use_gpu:
    gpu_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def train_weights_image_plus_spatial_classification():
    # need to train convolutional weights to classify images
    # and to infer the spatial location of (colored) saccade targets
    # saccade target locations can be inferred from 1st and/or 2nd convolutional layers
    # Reset TensorFlow graph

    tf.reset_default_graph()

    # Create placeholders for the model
    input_data   = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'input')
    target_data  = tf.placeholder(tf.float32, [par['batch_size'], dense_layers[-1]], 'target')
    loctarget_data  = tf.placeholder(tf.float32, [par['batch_size'], dense_layers_spatial[-1]], 'loctarget_data')

    # pass input through convolutional layers
    x,x_sploc = apply_convolutional_layers(input_data, None)
    print('x', x)

    # pass input through dense layers
    with tf.variable_scope('dense_layers'):
        c = 0.1
        W0 = tf.get_variable('W0', initializer = tf.random_uniform([dense_layers[0], dense_layers[1]], -c, c), trainable = True)
        W1 = tf.get_variable('W1', initializer = tf.random_uniform([dense_layers[1], dense_layers[2]], -c, c), trainable = True)
        b0 = tf.get_variable('b0', initializer = tf.zeros([1, dense_layers[1]]), trainable = True)
        b1 = tf.get_variable('b1', initializer = tf.zeros([1, dense_layers[2]]), trainable = True)

        W0_spatial = tf.get_variable('W0_spatial', initializer = tf.random_uniform([dense_layers_spatial[0], dense_layers_spatial[1]], -c, c), trainable = True)
        b0_spatial = tf.get_variable('b0_spatial', initializer = tf.zeros([1, dense_layers_spatial[1]]), trainable = True)

    x = tf.nn.relu(tf.matmul(x, W0) + b0)
    y = tf.matmul(x, W1) + b1

    y_sploc = tf.matmul(x_sploc, W0_spatial) + b0_spatial

    id_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = target_data, dim = 1))
    loc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_sploc, labels = loctarget_data, dim = 1))
    loss=id_loss+loc_loss
    optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
    train_op = optimizer.minimize(loss)

    # we will train the network on imagenet dataset
    stim = task.Stimulus()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(training_iterations):

            batch_data, batch_labels, spatial_labels = stim.generate_image_plus_spatial_batch()
            _, train_loss, ID_loss, sp_loss, y_hat, y_hat_sploc  = sess.run([train_op, loss, id_loss, loc_loss, y, y_sploc], \
                feed_dict = {input_data: batch_data, target_data: batch_labels, loctarget_data: spatial_labels})

            accuracy = np.sum(np.float32(np.argmax(y_hat,axis=1) == np.nonzero(batch_labels)[1]))/par['batch_size']
            acc_spatial = np.sum(np.float32(np.argmax(y_hat_sploc,axis=1) == np.nonzero(spatial_labels)[1]))/par['batch_size']
            if i%1000 == 0:
                print('Iteration ', i, ' Loss ', train_loss, 'ID_loss', ID_loss, 'Spatial_loss', sp_loss, 'Accuracy',accuracy,'Accuracy_spatial',acc_spatial)

        W = {}
        for var in tf.trainable_variables():
            W[var.op.name] = var.eval()

        pickle.dump(W, open(par['conv_weight_fn'],'wb'))
        print('Convolutional weights saved in ', par['conv_weight_fn'])

def train_weights_image_classification():

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    input_data   = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'input')
    target_data  = tf.placeholder(tf.float32, [par['batch_size'], dense_layers[-1]], 'target')

    # pass input through convolutional layers
    x, _ = apply_convolutional_layers(input_data, None)
    print('x', x)

    # pass input through dense layers
    with tf.variable_scope('dense_layers'):
        c = 0.1
        W0 = tf.get_variable('W0', initializer = tf.random_uniform([dense_layers[0], dense_layers[1]], -c, c), trainable = True)
        W1 = tf.get_variable('W1', initializer = tf.random_uniform([dense_layers[1], dense_layers[2]], -c, c), trainable = True)
        b0 = tf.get_variable('b0', initializer = tf.zeros([1, dense_layers[1]]), trainable = True)
        b1 = tf.get_variable('b1', initializer = tf.zeros([1, dense_layers[2]]), trainable = True)

    x = tf.nn.relu(tf.matmul(x, W0) + b0)
    y = tf.matmul(x, W1) + b1

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = target_data, dim = 1))
    optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
    train_op = optimizer.minimize(loss)

    # we will train the network on imagenet dataset
    stim = task.Stimulus()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(training_iterations):

            batch_data, batch_labels = stim.generate_image_batch()
            _, train_loss  = sess.run([train_op, loss], feed_dict = {input_data: batch_data, target_data: batch_labels})

            if i%1000 == 0:
                print('Iteration ', i, ' Loss ', train_loss)

        W = {}
        for var in tf.trainable_variables():
            W[var.op.name] = var.eval()

        pickle.dump(W, open(par['conv_weight_fn'],'wb'))
        print('Convolutional weights saved in ', par['conv_weight_fn'])


def apply_convolutional_layers(x, saved_weights_file):

    # load previous weights is saved_weights_file is not None,
    # otherwise, train new weights
    if saved_weights_file is None:
        kernel_init = [None for _ in range(num_layers)]
        bias_init = [tf.zeros_initializer() for _ in range(num_layers)]
        train = True
    else:
        kernel_names = ['conv2d']
        for i in range(1, num_layers):
            kernel_names.append('conv2d_' + str(i))
        conv_weights = pickle.load(open(saved_weights_file,'rb'))
        kernel_init = [tf.constant_initializer(conv_weights[k + '/kernel']) for k in kernel_names]
        bias_init = [tf.constant_initializer(conv_weights[k + '/bias']) for k in kernel_names]
        train = False

    for i in range(num_layers):

        x = tf.layers.conv2d(inputs = x, filters = par['conv_filters'][i], kernel_size = par['kernel_size'], kernel_initializer = kernel_init[i],  \
            bias_initializer = bias_init[i], strides = par['stride'], activation = tf.nn.relu, padding = 'SAME', trainable = train)

        if i==0:
            x_sploc = x

        if i > 0 and i%2 == 1:
            # apply max pooling and dropout after every second layer
            x = tf.layers.max_pooling2d(inputs = x, pool_size = par['pool_size'], strides = 2, padding='SAME')
            x = tf.nn.dropout(x, par['drop_keep_pct'])


    return tf.reshape(x, [par['batch_size'], -1]), tf.reshape(x_sploc, [par['batch_size'], -1])
