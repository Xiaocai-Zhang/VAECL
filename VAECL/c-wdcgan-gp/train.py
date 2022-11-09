# C-WDCGAN-GP
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import *

import argparse
import tensorflow as tf
import numpy as np

import cv2
from tensorflow import keras
import random
import matplotlib.pyplot as plt


gpu = True
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type = float,
    default = 5e-4, help = 'initial learning rate')
parser.add_argument('--gp_lambda', type = float,
    default = 20, help = 'lambda of gradient penalty')
parser.add_argument('--n_epoch', type = int,
    default = 40000, help = 'max # of epoch')
parser.add_argument('--n_update_dis', type = int,
    default = 5, help = '# of updates of discriminator per update of generator')
parser.add_argument('--noise_dim', type = int,
    default = 128, help = 'dimension of random noise')
parser.add_argument('--batch_size', type = int,
    default = 32, help = '# of batch size')


class config:
    save_tanker_target_path = '../DVTR/UAV-view/TA/'
    save_container_target_path = '../DVTR/UAV-view/CS/'
    save_bulkcarrier_target_path = '../DVTR/UAV-view/BC/'
    save_generalcargo_target_path = '../DVTR/UAV-view/GC/'


def samplefile(file_li):
    l = list(range(len(file_li)))
    kp_idx = random.sample(l, 50)
    kp_idx.sort()
    file_li_ = [file_li[idx] for idx in kp_idx]
    return file_li_

def channel4to3(img):
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def Dataset(args):
    # tanker class
    tanker_dir = os.listdir(config.save_tanker_target_path)
    tanker_dir = samplefile(tanker_dir)
    all_digits_tanker = []
    all_labels_tanker = []
    for item in tanker_dir:
        img = cv2.imread(config.save_tanker_target_path + item, cv2.IMREAD_UNCHANGED)
        img = channel4to3(img)
        all_digits_tanker.append(img)
        all_labels_tanker.append(0)
    all_digits_tanker = np.array(all_digits_tanker)
    all_labels_tanker = np.array(all_labels_tanker)

    # container class
    container_dir = os.listdir(config.save_container_target_path)
    container_dir = samplefile(container_dir)
    all_digits_container = []
    all_labels_container = []
    for item in container_dir:
        img = cv2.imread(config.save_container_target_path + item, cv2.IMREAD_UNCHANGED)
        img = channel4to3(img)
        all_digits_container.append(img)
        all_labels_container.append(1)
    all_digits_container = np.array(all_digits_container)
    all_labels_container = np.array(all_labels_container)

    # bulkcarrier class
    bulkcarrier_dir = os.listdir(config.save_bulkcarrier_target_path)
    bulkcarrier_dir = samplefile(bulkcarrier_dir)
    all_digits_bulkcarrier = []
    all_labels_bulkcarrier = []
    for item in bulkcarrier_dir:
        img = cv2.imread(config.save_bulkcarrier_target_path + item, cv2.IMREAD_UNCHANGED)
        img = channel4to3(img)
        all_digits_bulkcarrier.append(img)
        all_labels_bulkcarrier.append(2)
    all_digits_bulkcarrier = np.array(all_digits_bulkcarrier)
    all_labels_bulkcarrier = np.array(all_labels_bulkcarrier)

    # general cargo class
    generalcargo_dir = os.listdir(config.save_generalcargo_target_path)
    generalcargo_dir = samplefile(generalcargo_dir)
    all_digits_generalcargo = []
    all_labels_generalcargo = []
    for item in generalcargo_dir:
        img = cv2.imread(config.save_generalcargo_target_path + item, cv2.IMREAD_UNCHANGED)
        img = channel4to3(img)
        all_digits_generalcargo.append(img)
        all_labels_generalcargo.append(3)
    all_digits_generalcargo = np.array(all_digits_generalcargo)
    all_labels_generalcargo = np.array(all_labels_generalcargo)

    all_digits = np.concatenate([all_digits_tanker,all_digits_container,all_digits_bulkcarrier,all_digits_generalcargo],axis=0)
    all_digits = (all_digits.astype("float32")/ 255.0)*2-1

    all_labels = np.concatenate([all_labels_tanker, all_labels_container,all_labels_bulkcarrier,all_labels_generalcargo], axis=0)
    all_labels = keras.utils.to_categorical(all_labels, 4)
    all_labels = all_labels*2-1
    print('input shape: ',all_digits.shape)
    print('label shape: ', all_labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((all_digits,all_labels)).shuffle(250).batch(args.batch_size)

    return dataset

def recover(image):
    image_ = image.numpy()
    image__ = np.round(0.5*(image_+1)*255).astype(np.int32)
    return image__

def plot(fake_sample, epoch):
    plt.figure(figsize=(8, 8))
    for i in range(fake_sample.shape[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(fake_sample[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./plot/UAV-view/epoch_%d.png' % epoch)
    return None

def train_step_gen(args, one_hot_labels):
    batch_size = one_hot_labels.get_shape().as_list()[0]
    with tf.GradientTape() as tape:
        noise = tf.random.uniform([batch_size, args.noise_dim], -1.0, 1.0)
        random_labels = tf.concat(
            [noise, one_hot_labels], axis=1
        )
        fake_sample = args.gen(random_labels)

        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[256 * 256]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, 256, 256, 4)
        )

        fake_sample = tf.concat([fake_sample, image_one_hot_labels], -1)

        fake_score = args.dis(fake_sample)
        loss = - tf.reduce_mean(fake_score)
    gradients = tape.gradient(loss, args.gen.trainable_variables)
    args.gen_opt.apply_gradients(zip(gradients, args.gen.trainable_variables))
    args.gen_loss(loss)

def train_step_dis(args, real_sample, one_hot_labels):
    batch_size = real_sample.get_shape().as_list()[0]
    with tf.GradientTape() as tape:
        noise = tf.random.uniform([batch_size, args.noise_dim], -1.0, 1.0)

        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[256 * 256]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, 256, 256, 4)
        )

        noise_label = tf.concat(
            [noise, one_hot_labels], axis=1
        )

        fake_sample = args.gen(noise_label)
        fake_sample = tf.concat([fake_sample, image_one_hot_labels], -1)
        real_sample = tf.concat([real_sample, image_one_hot_labels], -1)

        real_score = args.dis(real_sample)
        fake_score = args.dis(fake_sample)

        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        inter_sample = fake_sample * alpha + real_sample * (1 - alpha)
        with tf.GradientTape() as tape_gp:
            tape_gp.watch(inter_sample)
            inter_score = args.dis(inter_sample)
        gp_gradients = tape_gp.gradient(inter_score, inter_sample)
        gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis = [1, 2, 3]))
        gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

        loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp * args.gp_lambda

    gradients = tape.gradient(loss, args.dis.trainable_variables)
    args.dis_opt.apply_gradients(zip(gradients, args.dis.trainable_variables))

    args.dis_loss(loss)
    args.adv_loss(loss - gp * args.gp_lambda)

def test_step(args, epoch):
    noise = tf.random.uniform([64, args.noise_dim], -1.0, 1.0)
    one_hot_labels = np.eye(4)[np.random.choice(4, 64)]
    one_hot_labels = one_hot_labels * 2 - 1
    random_labels = tf.concat(
        [noise, one_hot_labels], axis=1
    )
    fake_sample = args.gen(random_labels)
    fake_sample = recover(fake_sample)
    plot(fake_sample, epoch)

def train(args):
    for epoch in range(1,args.n_epoch+1):
        cnt = 0
        for batch in args.ds:
            real_images, one_hot_labels = batch

            cnt += 1
            if cnt % (args.n_update_dis + 1) > 0:
                train_step_dis(args, real_images, one_hot_labels)
            else:
                train_step_gen(args, one_hot_labels)

        if epoch == 1 or epoch % 50 == 0:
            test_step(args, epoch)
            args.gen.save_weights("./save_gen/UAV-view/generator.h5")

        template = 'Epoch {}, Gen Loss: {}, Dis Loss: {}, Adv Loss: {}'
        print (template.format(epoch, args.gen_loss.result(),
                args.dis_loss.result(), args.adv_loss.result()))
        args.dis_loss.reset_states()
        args.adv_loss.reset_states()
        args.gen_loss.reset_states()

if __name__ == '__main__':
    args = parser.parse_args()
    args.ds = Dataset(args)

    # Initialize Networks
    args.gen = Generator()
    args.dis = Discriminator()

    # Initialize Optimizer
    args.gen_opt = tf.keras.optimizers.Adam(args.learning_rate)
    args.dis_opt = tf.keras.optimizers.Adam(args.learning_rate)

    # Initialize Metrics
    args.adv_loss = tf.keras.metrics.Mean(name = 'Adversarial_Loss')
    args.gen_loss = tf.keras.metrics.Mean(name = 'Generator_Loss')
    args.dis_loss = tf.keras.metrics.Mean(name = 'Discriminator_Loss')

    train(args)
