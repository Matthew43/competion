import os
import tensorflow as tf
import matplotlib.pyplot as plt

image_dir = 'E:\\train'
id_map_label = 'E:\\大数据\\百度\\训练数据\\data_train_image.txt'


def get_image_and_label():
    id2label = {}
    with open(id_map_label, 'r') as f:
        for line in f.readlines():
            temp = line.split(' ')
            id2label[temp[0]] = temp[1].strip()

    images = []
    labels = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # print(file)
            id = file.split('.')[0]
            image = os.path.join(root, file)
            images.append(image)
            labels.append(int(id2label[id]))
    # for i,l in zip(images,labels):
    #     print(i,l)
    # print(len(images),' ',len(labels) )

    return images, labels


def get_batch(images, labels, image_W, image_H, batch_size, capacity):
    '''
    :param labels: list type
    :param image_W: image width
    :param image_H: mage height
    :param batch_size:  batch size
    :param capacity: the maximum elements in queue
    :return:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    images = tf.cast(images, tf.string)
    # print(labels)
    labels = tf.cast(labels, tf.int32)

    input_queue = tf.train.slice_input_producer([images, labels])
    labels = input_queue[1]
    images = tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3)

    # change image size
    images = tf.image.resize_image_with_crop_or_pad(images, image_W, image_H)

    images = tf.image.per_image_standardization(images)

    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity,min_after_dequeue=capacity-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return label_batch,image_batch

if __name__ == '__main__':
    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208
    images, labels = get_image_and_label()
    label_batch, image_batch = get_batch(images,labels,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1:

                img, label = sess.run([image_batch, label_batch])

                # just test one batch
                for j in range(BATCH_SIZE):
                    print('label: %d' %label[j])
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i+=1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

