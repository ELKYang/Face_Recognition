import numpy as np
import tensorflow as tf
import glob
import os
from skimage import io

path = '/openbayes/input/input0'

#读取数据
def get_image(path):
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    images = []
    labels = []
    for idx, folder in enumerate(cate):
        for image in glob.glob(folder + '/*.png'):
            imagedata = io.imread(image)
            images.append(imagedata)
            labels.append(idx)
    np.save("path_num.npy",cate)
    return np.asarray(images, np.float32), np.asarray(labels, np.int32)
ImageData, ImageLable = get_image(path)
np.savetxt("Label.txt",ImageLable)

#打乱数据
state = np.random.get_state()
np.random.shuffle(ImageData)
np.random.set_state(state)
np.random.shuffle(ImageLable)

#将数据分为train和validation（7:3）
Data_num = ImageData.shape[0]
ratio = 0.7
Truncated_num = np.int(Data_num * ratio)
Data_train = ImageData[:Truncated_num]
Lable_train = ImageLable[:Truncated_num]
Data_val = ImageData[Truncated_num:]
Lable_val = ImageLable[Truncated_num:]

with tf.name_scope('input'):
    Input_data = tf.placeholder(tf.float32,
                                shape=[None, 128, 128, 3],
                                name='Input_data')
    Output_label = tf.placeholder(tf.int32, shape=[None,], name='Output_label')
tf_is_training = tf.placeholder(tf.bool, None)
#建立网络结构
with tf.name_scope('conv1'):
    conv1 = tf.layers.conv2d(
        inputs=Input_data,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

with tf.name_scope('pool1'):
    pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

with tf.name_scope('conv2'):
    conv2=tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

with tf.name_scope('pool2'):
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

with tf.name_scope('conv3'):
    conv3=tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

with tf.name_scope('pool3'):
    pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=2)

with tf.name_scope('conv4'):
    conv4=tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

with tf.name_scope('pool4'):
    pool4=tf.layers.max_pooling2d(inputs=conv4,pool_size=[2,2],strides=2)

with tf.name_scope('pool4_flat'):
    pool4_flat = tf.reshape(pool4, [-1, 8 * 8 * 128])

with tf.name_scope('dense1'):
    dense1 = tf.layers.dense(inputs=pool4_flat, 
                        units=1024, 
                        activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

with tf.name_scope('dropout1'):
    dropout1=tf.layers.dropout(dense1, rate=0.5, training=tf_is_training)

with tf.name_scope('dense2'):
    dense2= tf.layers.dense(inputs=dropout1, 
                        units=512, 
                        activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

with tf.name_scope('dropout2'):
    dropout2=tf.layers.dropout(dense2, rate=0.5, training=tf_is_training)

with tf.name_scope('logits'):
    logits= tf.layers.dense(inputs=dropout2, 
                            units=84,
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

tf.add_to_collection('pred_network', logits)
with tf.name_scope('cross_entropy'):
    loss=tf.losses.sparse_softmax_cross_entropy(labels=Output_label,logits=logits)
    tf.summary.scalar("loss", loss)
with tf.name_scope('train'):
    train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), Output_label)  
    with tf.name_scope('accuracy'):
        acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("acc", acc)

#批处理
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#开始训练
saver=tf.train.Saver(max_to_keep=3)
max_acc=0
n_epoch=60
batch_size=32
train_num=0
val_num=0
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
merged_summary_op = tf.summary.merge_all()
summary_writer_train = tf.summary.FileWriter('./tf_dir/train', graph=tf.get_default_graph())
summary_writer_val=tf.summary.FileWriter('./tf_dir/val')
for epoch in range(n_epoch):
    #training
    train_loss, train_acc, n_batch_train = 0, 0, 0
    for x_train_a, y_train_a in minibatches(Data_train, Lable_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={Input_data: x_train_a, Output_label: y_train_a,tf_is_training: True})
        result=sess.run(merged_summary_op,feed_dict={Input_data: x_train_a, Output_label: y_train_a,tf_is_training: True})
        summary_writer_train.add_summary(result,train_num)
        train_loss += err; train_acc += ac; n_batch_train += 1;train_num+=1
    #validation
    val_loss, val_acc, n_batch_val = 0, 0, 0
    for x_val_a, y_val_a in minibatches(Data_val, Lable_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={Input_data: x_val_a, Output_label: y_val_a,tf_is_training: False})
        result=sess.run(merged_summary_op,feed_dict={Input_data: x_val_a, Output_label: y_val_a,tf_is_training: False})
        summary_writer_val.add_summary(result,val_num)
        val_loss += err; val_acc += ac; n_batch_val += 1;val_num += 1
    if val_acc>max_acc:
        max_acc=val_acc
        saver.save(sess,'./log/model',global_step=epoch+1)
summary_writer.close()
sess.close()