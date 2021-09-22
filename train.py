from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from model.utils import *
from model.models import cengcn


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'facebook_ego', 'Dataset string.')  
flags.DEFINE_string('model', 'cengcn', 'Model string.') 
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs',1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('ckpt', '_degree', 'save sess.')

percent=0.05
p=1
q=-1

if FLAGS.dataset=='gplus_ego':
    percent=0.03
    p=1.25
    q=-0.75
if FLAGS.dataset=='facebook_ego':
    percent=0.05
    p=1
    q=-1
if FLAGS.dataset=='twitter_ego':
    percent=0.001
    p=1
    q=-1
if FLAGS.dataset=='youtube':
    percent=0.03
    p=0.25
    q=-0.5
if FLAGS.dataset=='livejournal':
    percent=0.001
    p=2.5
    q=-2

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_new(FLAGS.dataset)

FLAGS.ckpt=FLAGS.dataset+FLAGS.ckpt

# Some preprocessing
features = preprocess_features(features)
if FLAGS.pattern == "degree":
    tuple_ele,center_node=preprocess_degree(adj,percent,p,q)
elif FLAGS.pattern == "eig":
    w,v=np.linalg.eig(adj.toarray())
    tuple_ele,center_node=preprocess_eig(adj,w,v,percent,p,q)



tran_adj=adj.toarray()
for c in center_node:
    tran_adj[c,:]=np.array([0]*adj.shape[0])
    tran_adj[:,c]=np.array([0]*adj.shape[0])

tran_adj+=np.eye(adj.shape[0],dtype=np.float32)
tran_adj=(1-tran_adj)*(-1e10)

if FLAGS.model == 'cengcn':
    support = [tuple_ele]
    num_supports = 1
    model_func = cengcn

else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], adj=tran_adj,logging=True)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders,is_test=False):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    if is_test:
        outs_val = sess.run([model.loss, model.accuracy,model.predict()], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2],(time.time() - t_test)
    else:
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1],(time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
accs=[]
max_acc=0.0
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    # accs.append(acc)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    if max_acc<acc:
        max_acc=acc
        test_cost, test_acc, pred, test_duration = evaluate(features, support, y_test, test_mask, placeholders,
                                                            is_test=True)
        # model.save(sess)

print("Optimization Finished!")

# model.load(sess)
# Testing
# test_cost, test_acc, pred,test_duration = evaluate(features, support, y_test, test_mask, placeholders,is_test=True)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

sess.close()
