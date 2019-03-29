import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import tensorflow as tf
import os 
import cv2
# from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline

# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
file_id = ''
downloaded = drive.CreateFile({'id': file_id})
print('Downloaded content "{}"'.format(downloaded.GetContentFile('cell-images-for-detecting-malaria.zip')))

!unzip cell-images-for-detecting-malaria.zip

train_data = []
train_label = []
test_data = []
test_label=[]

path1 = "cell_images/Parasitized"
path2 = "cell_images/Uninfected"

count1 = 0
for files in os.listdir(path1):
    
    if (files.endswith(".png")):
        img = cv2.imread(os.path.join(path1,files))
        img = cv2.resize(img, (28, 28)) 
        train_data.append(img)
        train_label.append([1,0])
        count1 += 1
        if count1 >= 10000:
            img = cv2.imread(os.path.join(path1,files))
            img = cv2.resize(img, (28, 28)) 
            test_data.append(img)
            test_label.append([1,0])

count2 = 0
for files in os.listdir(path2):
    
    if (files.endswith(".png")):
        img = cv2.imread(os.path.join(path2,files))
        img = cv2.resize(img, (28, 28)) 
        train_data.append(img)
        train_label.append([0,1])
        count2 += 1
        if count2 >= 10000:
            img = cv2.imread(os.path.join(path2,files))
            img = cv2.resize(img, (28, 28)) 
            test_data.append(img)            
            test_label.append([0,1])

from sklearn.utils import shuffle
train_data,train_label = shuffle(train_data,train_label,random_state = 0)
test_data,test_label = shuffle(test_data,test_label,random_state=1)

train_data[0].shape

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(test_data[3], cmap='gray')

training_iters = 200 
learning_rate = 0.001 
batch_size = 128

# MNIST data input (img shape: 28*28)
n_input = 28

# MNIST total classes (0-9 digits)
n_classes = 2

train_data = np.asarray(train_data)
test_data = np.asarray(test_data)

# Reshape training and testing image
train_X = train_data.reshape(-1, 28, 28, 3)
test_X = test_data.reshape(-1,28,28,3)

train_X.shape, test_X.shape

train_label = np.asarray(train_label)
test_label = np.asarray(test_label)

# Reshape training and testing image
train_y = train_label
test_y = test_label

train_y.shape, test_y.shape

#both placeholders are of type float
x = tf.placeholder("float", [None, 28,28,3])
y = tf.placeholder("float", [None,2])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

weights = {
    'wc1': tf.get_variable('W40', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W41', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W42', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W43', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W46', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B40', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B41', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B42', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B43', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B44', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)
    print(conv1)
    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    print(conv2)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)
    print(conv3)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    print(fc1)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print(fc1)
    fc1 = tf.nn.relu(fc1)
    print(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()
