import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
import random



class config:
    '''
    define parameters & paths
    '''
    save_tanker_target_train_path = '../DVTR/UAV-view/train/TA/'
    save_tanker_target_test_path = '../DVTR/UAV-view/test/TA/'
    save_container_target_train_path = '../DVTR/UAV-view/train/CS/'
    save_container_target_test_path = '../DVTR/UAV-view/test/CS/'
    save_bulkcarrier_target_train_path = '../DVTR/UAV-view/train/BC/'
    save_bulkcarrier_target_test_path = '../DVTR/UAV-view/test/BC/'
    save_generalcargo_target_train_path = '../DVTR/UAV-view/train/GC/'
    save_generalcargo_target_test_path = '../DVTR/UAV-view/test/GC/'


# set seed
seed = 409
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)


class Dataset:
    def channel4to3(self,img):
        '''
        reduce 4 channel img to 3 channel
        :param img: 4 channel img
        :return: 3 channel img
        '''
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def rt_data_label(self,dir,label,path,reduce=True):
        '''

        :param dir: data directory
        :param label: data label
        :param path:
        :param reduce:
        :return:
        '''
        all_digits = []
        all_labels = []
        for item in dir:
            img = cv2.imread(path + item, cv2.IMREAD_UNCHANGED)
            if reduce:
                img = self.channel4to3(img)
            all_digits.append(img)
            all_labels.append(label)
        all_digits = np.array(all_digits)
        all_labels = np.array(all_labels)
        return all_digits,all_labels

    def uav_view_dataset(self):
        '''
        get UAV-view image data ready
        :return: data & labels
        '''
        # tanker class
        tanker_dir = os.listdir(config.save_tanker_target_train_path)
        tanker_dir_rest = os.listdir(config.save_tanker_target_test_path)
        all_digits_tanker,all_labels_tanker=self.rt_data_label(tanker_dir,0,config.save_tanker_target_train_path,reduce=False)
        all_digits_tanker_rest, all_labels_tanker_rest = self.rt_data_label(tanker_dir_rest, 0, config.save_tanker_target_test_path,reduce=False)

        # container class
        container_dir = os.listdir(config.save_container_target_train_path)
        container_dir_rest = os.listdir(config.save_container_target_test_path)
        all_digits_container, all_labels_container = self.rt_data_label(container_dir, 1, config.save_container_target_train_path,reduce=True)
        all_digits_container_rest, all_labels_container_rest = self.rt_data_label(container_dir_rest, 1,
                                                                            config.save_container_target_test_path,reduce=True)

        # bulkcarrier class
        bulkcarrier_dir = os.listdir(config.save_bulkcarrier_target_train_path)
        bulkcarrier_dir_rest = os.listdir(config.save_bulkcarrier_target_test_path)
        all_digits_bulkcarrier, all_labels_bulkcarrier = self.rt_data_label(bulkcarrier_dir, 2,
                                                                        config.save_bulkcarrier_target_train_path,reduce=False)
        all_digits_bulkcarrier_rest, all_labels_bulkcarrier_rest = self.rt_data_label(bulkcarrier_dir_rest, 2,
                                                                                  config.save_bulkcarrier_target_test_path,reduce=False)

        # general cargo class
        generalcargo_dir = os.listdir(config.save_generalcargo_target_train_path)
        generalcargo_dir_rest = os.listdir(config.save_generalcargo_target_test_path)
        all_digits_generalcargo, all_labels_generalcargo = self.rt_data_label(generalcargo_dir, 3,
                                                                            config.save_generalcargo_target_train_path,reduce=True)
        all_digits_generalcargo_rest, all_labels_generalcargo_rest = self.rt_data_label(generalcargo_dir_rest, 3,
                                                                                      config.save_generalcargo_target_test_path,reduce=True)

        all_digits = np.concatenate(
            [all_digits_tanker, all_digits_container, all_digits_bulkcarrier, all_digits_generalcargo], axis=0)
        all_digits = (all_digits.astype("float32") / 255.0) * 2 - 1

        all_digits_rest = np.concatenate(
            [all_digits_tanker_rest, all_digits_container_rest, all_digits_bulkcarrier_rest, all_digits_generalcargo_rest], axis=0)
        all_digits_rest = (all_digits_rest.astype("float32") / 255.0) * 2 - 1

        all_labels = np.concatenate(
            [all_labels_tanker, all_labels_container, all_labels_bulkcarrier, all_labels_generalcargo], axis=0)
        all_labels = keras.utils.to_categorical(all_labels, 4)

        all_labels_rest = np.concatenate(
            [all_labels_tanker_rest, all_labels_container_rest, all_labels_bulkcarrier_rest, all_labels_generalcargo_rest], axis=0)
        all_labels_rest = keras.utils.to_categorical(all_labels_rest, 4)

        return all_digits,all_labels,all_digits_rest,all_labels_rest
