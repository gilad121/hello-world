import pickle
import numpy as np
import os
import urllib
import tarfile
import zipfile
import sys
import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np
from time import time
import math

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
  

def eval_set_augmentation(image):
	global _CROPPED_IMAGE_SIZE
    height = _CROPPED_IMAGE_SIZE
	width = _CROPPED_IMAGE_SIZE
	
	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    standartized_image = tf.image.per_image_standardization(resized_image)
	return standartized_image
	

def train_set_augmentation(image):
    global _CROPPED_IMAGE_SIZE
    height = _CROPPED_IMAGE_SIZE
	width = _CROPPED_IMAGE_SIZE
	
    distorted_image = tf.random_crop(image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
	distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
	distorted_image = tf.image.per_image_standardization(distorted_image)
	return distorted_image
	
  
def model():    
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
    y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
    x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
    keep_prob2 = tf.placeholder(tf.float32 , name = "dropout parameter")    
	
	run_evaluation = tf.placeholder(tf.float32)
	if run_evaluation is 1:
	    x_image = map_fn(eval_set_augmentation, x_image)
	else:
	    x_image = map_fn(train_set_augmentation, x_image)
	

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)


    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 32],
                                             stddev=5e-2,
                                             wd=None)
											 
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
		
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')


    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 32],
                                             stddev=5e-2,
                                             wd=None)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
	
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
					
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fc3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool3, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        print("dim of flatted feature vector is = ", str(dim)) # DEBUG
        weights = _variable_with_weight_decay('weights', shape=[dim, 70],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [70], tf.constant_initializer(0.1))
        fc3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		
    with tf.variable_scope('fc4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[70, 50],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [50], tf.constant_initializer(0.1))
        fc4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		
	drop5 = tf.nn.dropout(fc4, keep_prob2)

		
  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  # with tf.variable_scope('softmax_linear') as scope:
  #  weights = _variable_with_weight_decay('weights', [50, NUM_CLASSES],
  #                                        stddev=1/50.0, wd=None)
  #  biases = _variable_on_cpu('biases', [NUM_CLASSES],
  #                            tf.constant_initializer(0.0))
  #  softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
  #  _activation_summary(softmax_linear)

    softmax = tf.nn.softmax(tf.layers.dense(inputs=drop5, units=_NUM_CLASSES))

    y_pred_cls = tf.argmax(softmax, axis=1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))
     
    optimizer = tf.train.AdamOptimizer(1e-4  , beta1=0.9,   
                                beta2=0.999,
                               epsilon=1e-08).minimize(loss)

    # PREDICTION AND ACCURACY CALCULATION
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
    return x, y, loss ,optimizer , correct_prediction ,accuracy, y_pred_cls, keep_prob2, run_evaluation


def get_data_set(name="train"):
    x = None
    y = None

    maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f)
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    return x, dense_to_one_hot(y)


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urllib.urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)
		
		
def train(epoch):
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
	
	# Calculate mean of all batch_acc, batch_loss
	epoch_acc = 0
	epoch_loss = 0
	
    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]

        start_time = time()
        _,batch_loss, batch_acc = sess.run(
            [ optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys ,keep_prob2:0.5, run_evaluation = 0})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))
            msg = "step: {} , batch_acc = {} , batch loss = {}"
            print(msg.format(s, batch_acc, batch_loss))
		
		epoch_acc = epoch_acc + ((batch_acc - epoch_acc) / (s + 1))
		epoch_loss = epoch_loss + ((batch_loss - epoch_loss) / (s + 1))
	
	train_acc[epoch] = epoch_acc
	train_loss[epoch] = epoch_loss
	msg = "epoch: {} , train accuracy = {} , train loss = {}"
	print(msg.format(epoch, epoch_acc, epoch_loss))		# DEBUG

    test_and_save(epoch)

def test_and_save(epoch):
    global global_accuracy
	
	# Calculate mean of all batch_acc, batch_loss
	epoch_loss = 0
	
    i = 0
	iter_num = 0	# iteration number
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j], batch_loss = sess.run(
            y_pred_cls, loss
            feed_dict={x: batch_xs, y: batch_ys ,keep_prob2:1, run_evaluation = 1}
        )
        i = j
		epoch_loss = epoch_loss + ((batch_loss - epoch_loss) / (iter_num + 1))
		iter_num = iter_num + 1

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
	
	test_acc[epoch] = acc
	test_loss[epoch] = epoch_loss

    mes = "\nEpoch {} - test accuracy: {:.2f}% ({}/{}), test loss = {:.3f}"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x)), epoch_loss)

    if global_accuracy != 0 and global_accuracy < acc:
        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")

	
# switch from accuracy to error
def plot_graphs():  
	# accuracy(or error) - train, test
	sub1 = plt.subplot(2, 1, 1)
	sub1.plot(train_acc)
	sub1.plot(test_acc)
	sub1.set_title('accuracy', fontsize=20)
	# plt.xlabel('epoch', fontsize=18)
	sub1.ylabel('accuracy', fontsize=16)
	
	# loss - train, test
	sub2 = plt.subplot(2, 1, 2)
	sub2.plot(train_loss)
	sub2.plot(test_loss)
	sub2.set_title('loss', fontsize=20)
	sub2.xlabel('epoch', fontsize=18)
	sub2.ylabel('loss', fontsize=16)
	
	plt.show()
	
def load_model_and_run(path)
    saver.restore(sess, path)
	for epoch in range(_EPOCH):
	    test_and_save(epoch)
		
	mean_acc = np.mean(test_acc)
	print("Mean accuracy on test set = {}".format(mean_acc))
    
	
tf.reset_default_graph()
sess = tf.Session()

saver = tf.train.Saver()

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y,loss, optimizer , correct_prediction ,accuracy, y_pred_cls ,keep_prob2, run_evaluation= model()
global_accuracy = 0

# PARAMS
_CROPPED_IMAGE_SIZE = 24
_BATCH_SIZE = 128
_EPOCH = 300

# test & train - accuracy & loss
train_acc = np.zeros(shape=(_EPOCH,))
train_loss = np.zeros(shape=(_EPOCH,))
test_acc = np.zeros(shape=(_EPOCH,))
test_loss = np.zeros(shape=(_EPOCH,))

sess.run(tf.global_variables_initializer())

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print(total_parameters)
raw_input()

def main():
    for i in range(_EPOCH):
        print("\nEpoch: {0}/{1}\n".format((i+1), _EPOCH))
        train(i)
        save_path = saver.save(sess, "/tmp/model.ckpt")
	plot_graphs()


if __name__ == "__main__":
    main()