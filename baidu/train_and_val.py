import tensorflow as tf
import os
import VGG
import input
import tools

IMG_W = 300
IMG_H = 300
N_CLASSES = 100
BATCH_SIZE = 2
learning_rate = 0.01
MAX_STEP = 20000
IS_PRETRAIN = True

CAPACITY = 2000
MIN_AFTER_DEQUEUE = 300

train_image_dir = 'E:/python_study/data/train'
train_id_map_label_file = 'new_train.txt'

val_image_dir = 'E:/python_study/data/test1'
# val_id_map_label_file = "‪E:/python_study/data/val.txt"
val_id_map_label_file = "new_val.txt"


def train():
    pre_trained_weights = 'E:/model/vgg16.npy'
    # data_dir = ''
    train_log_dir = 'E:/python_study/data/log/train'
    val_log_dir = 'E:/python_study/data/log/val'

    with tf.name_scope('input'):
        train_images, train_labels = input.get_image_and_label(train_image_dir, train_id_map_label_file)
        train_image_batch, train_label_batch = input.get_batch(train_images, train_labels, IMG_W, IMG_H, BATCH_SIZE,
                                                               N_CLASSES,
                                                               CAPACITY, MIN_AFTER_DEQUEUE)

        val_images, val_labels = input.get_image_and_label(val_image_dir, val_id_map_label_file)
        val_image_batch, val_label_batch = input.get_batch(val_images, val_labels, IMG_W, IMG_H, BATCH_SIZE, N_CLASSES,
                                                           CAPACITY, MIN_AFTER_DEQUEUE)
    logits = VGG.VGG16N(train_image_batch, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, train_label_batch)
    accuracy = tools.accuracy(logits, train_label_batch)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # load weight
        tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_image_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                                feed_dict={x: tra_images, y_: tra_labels})

                # do print per 50 times
                if step % 50 == 0 or (step + 1) == MAX_STEP:
                    print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op)
                    tra_summary_writer.add_summary(summary_str, step)

                # do validate per 200 times
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, accuracy],
                                                 feed_dict={x: val_images, y_: val_labels})
                    print('(validate) Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))

                    summary_str = sess.run(summary_op)
                    val_summary_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_log_dir, 'model{}.ckpt'.format(step))
                    saver.save(sess, checkpoint_path, global_step=step)



        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

def continue_train():
    log_dir = 'F:/log/train'
    continue_train_log_dir = 'E:/python_study/data/log/continue_train'
    train_save_dir = 'F:/log/save'

    with tf.name_scope('input'):
        train_images, train_labels = input.get_image_and_label(train_image_dir, train_id_map_label_file)
        train_image_batch, train_label_batch = input.get_batch(train_images, train_labels, IMG_W, IMG_H, BATCH_SIZE,
                                                               N_CLASSES,
                                                               CAPACITY, MIN_AFTER_DEQUEUE)

    logits = VGG.VGG16N(train_image_batch, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, train_label_batch)
    accuracy = tools.accuracy(logits, train_label_batch)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # sess.run(init)
        # load pre-trained model
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.all_model_checkpoint_paths[2]:
            global_step = ckpt.all_model_checkpoint_paths[2].split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.all_model_checkpoint_paths[2])
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tra_summary_writer = tf.summary.FileWriter(continue_train_log_dir, sess.graph)

        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_image_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                                feed_dict={x: tra_images, y_: tra_labels})

                # do print per 50 times
                if step % 50 == 0 or (step + 1) == MAX_STEP:
                    print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op)
                    tra_summary_writer.add_summary(summary_str, step)


                if step % 5000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_save_dir, 'model{}.ckpt'.format(step))
                    saver.save(sess, checkpoint_path, global_step=step)



        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()







if __name__ == '__main__':
    # train()
    continue_train()