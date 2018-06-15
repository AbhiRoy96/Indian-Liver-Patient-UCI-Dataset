import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib import layers
import time

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pd.read_csv('C:/Users/dell/Desktop/ILPD.csv')


def label_encode(label):
  val=[]
  if label == 1:
    val = [1,0]
  elif label == 2:
    val = [0,1]
  return val

#training and test datasets
X = data[["AGE","GENDER","TB","DB","ALK","SGPT","SGOT","TP","ALB","AG"]]
y = data["LABEL"]

X_train, X_test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test = []
y_train = []

for y in y_Train:
  y_train.append(label_encode(y))

for y in y_Test:
  y_test.append(label_encode(y))


#neuralnetwork params
learning_rate = 0.01
training_epochs = 1000000
display_steps = 100000

n_input = 10
n_hidden = 40
n_output = 2

#placeholders and variables
X_tf = tf.placeholder("float", [None, n_input])
y_tf = tf.placeholder("float", [None, n_output])

#weights and bias
weights = {
    "hidden": tf.Variable(tf.random_normal([n_input, n_hidden])),
    "output": tf.Variable(tf.random_normal([n_hidden, n_output])),
}

bias = {
    "hidden": tf.Variable(tf.random_normal([n_hidden])),
    "output": tf.Variable(tf.random_normal([n_output])),
}


def model(X, weights, bias):
	layer1 = tf.add( tf.matmul(X, weights["hidden"]), bias["hidden"])
	layer1 = tf.nn.relu(layer1)
	
	output_layer = tf.matmul(layer1, weights["output"])+ bias["output"]
	return output_layer


#Define model
pred = model(X_tf, weights, bias) 

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_tf))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Initializing global variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
  	sess.run(init)

  	for epoch in range(training_epochs):
  		_, c = sess.run([optimizer, cost], feed_dict={X_tf: X_train, y_tf: y_train})
  		if(epoch + 1) % display_steps == 0:
  			print ("Epoch: " + str(epoch+1) + " Cost: " + str(c))

  	print("Optimization Finished!")

  	test_result = sess.run(pred, feed_dict={X_tf: X_train})
  	correct_pred = tf.equal(tf.argmax(test_result, 1), tf.argmax(y_train, 1))

  	accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
  	print ("Accuracy:c" + str(accuracy.eval({X_tf: X_test, y_tf: y_test})))