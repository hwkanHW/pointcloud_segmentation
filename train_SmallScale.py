import os
import argparse
import numpy as np
import tensorflow as tf
import open3d as o3d
import utils.mars_dataset as mars_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--use_color', type=bool, default=True, help='if use color information [default: True]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=3000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

USE_COLOR = FLAGS.use_color

print("-------------------------use color-----------------------")
print(USE_COLOR)
print("---------------------------------------------------------")

if USE_COLOR:
    import models.pointnet2_semrgb_seg_revised as MODEL
else:
    import models.pointnet2_sem_seg as MODEL

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "npy_file")
pcd_npy_file = os.path.join(DATA_DIR, "pts_data.npy")
cls_npy_file = os.path.join(DATA_DIR, "cls_data.npy")
cos_npy_file = os.path.join(DATA_DIR, "cos_data.npy")
npy_file = [pcd_npy_file, cls_npy_file, cos_npy_file]

dataset = mars_dataset.MarsDataset(npy_file=npy_file, npoints=NUM_POINT,
                                   batch_size=BATCH_SIZE, random_num=50, load_from_npy=True)


def log_string(out_str):
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, 2, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join("checkpoint", 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join("checkpoint", 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'merged': merged,
               'train_op': train_op,
               'step': batch,
               'end_points': end_points}

        # best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))

            train_one_epoch(sess, ops, train_writer)
            dataset.reset()

            eval_one_epoch(sess, ops, test_writer)
            dataset.reset()

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join("checkpoint", "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    ###
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while dataset.has_next_batch():
        batch_pts, batch_label, batch_colors = dataset.next_batch()
        batch_pts_norm = dataset.pts_normilize(batch_pts)
        if USE_COLOR:
            batch_data = np.zeros([BATCH_SIZE, NUM_POINT, 6])
            batch_data[:, :, :3] = batch_pts_norm
            batch_data[:, :, 3:] = batch_colors
        else:
            batch_data = batch_pts_norm
        ###
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val

        if (batch_idx + 1) % 10 == 0:
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx = batch_idx+1


def eval_one_epoch(sess, ops, test_writer):
    acc = 0
    idx = 0
    while dataset.has_next_batch():
        batch_pts, batch_label, batch_colors = dataset.next_batch()
        batch_pts_norm = dataset.pts_normilize(batch_pts)
        if USE_COLOR:
            batch_data = np.zeros([BATCH_SIZE, NUM_POINT, 6])
            batch_data[:, :, :3] = batch_pts_norm
            batch_data[:, :, 3:] = batch_colors
        else:
            batch_data = batch_pts_norm
        ###
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: False}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        cls_pred = np.argmax(pred_val,axis=-1)
        ##
        for i in range(len(cls_pred)):
            cls_gt_idx = np.zeros(batch_label[i].shape)
            cls_pred_idx = np.zeros(batch_label[i].shape)
            index_gt = np.where(batch_label[i]== 1)[0]
            index_pred = np.where(cls_pred[i] == 1)[0]
            cls_gt_idx[index_gt] = 1
            cls_pred_idx[index_pred] = 1
            intersection = np.sum(np.logical_and(cls_gt_idx, cls_pred_idx))
            union = np.sum(np.logical_or(cls_gt_idx, cls_pred_idx))
            acc = acc + (intersection/union)
            idx = idx+1
    print("IoU on Apple segmentation %s"%(acc/idx))




if __name__ == "__main__":
    tf.test.is_gpu_available()
    tf.test.is_gpu_available()
    train()

