import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np



class config:
    '''
    define parameters & paths
    '''
    size_per_class = 50*20
    noise_dim = 128
    uav_view_generator = '../c-wdcgan-gp/save_gen/UAV-view/generator.h5'


def build_model():
    '''
    build the generator architecture
    :return: model of generator
    '''
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(132,)))
    model.add(tf.keras.layers.Dense(8 * 8 * 256))
    model.add(tf.keras.layers.Reshape((8, 8, 256)))
    model.add(tf.keras.layers.Conv2DTranspose(256, 5, 2, activation=tf.nn.relu, padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(128, 5, 2, activation=tf.nn.relu, padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(64, 5, 2, activation=tf.nn.relu, padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(32, 5, 2, activation=tf.nn.relu, padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(16, 5, 2, activation=tf.nn.relu, padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(3, 3, 1, activation=tf.nn.tanh, padding='same'))

    # you can either compile or not the model
    model.compile()
    return model


class generate_data:
    def gen_labels(self,shiptype,size):
        '''
        functio to generate labels for given type of vessels
        :param shiptype: ship type
        :param size: sample size
        :return: array of label
        '''
        if shiptype=='tanker':
            a = np.array([1, -1, -1, -1])
            b = np.tile(a, size)
            b = np.reshape(b, newshape=(size, 4))
            return b
        elif shiptype=='container':
            a = np.array([-1, 1, -1, -1])
            b = np.tile(a, size)
            b = np.reshape(b, newshape=(size, 4))
            return b
        elif shiptype=='bulkcarrier':
            a = np.array([-1, -1, 1, -1])
            b = np.tile(a, size)
            b = np.reshape(b, newshape=(size, 4))
            return b
        elif shiptype=='general cargo':
            a = np.array([-1, -1, -1, 1])
            b = np.tile(a, size)
            b = np.reshape(b, newshape=(size, 4))
            return b

    def uav_view_sample(self):
        '''
        function to generate UAV-view images via trained GAN model
        :return: data and label
        '''
        generator = build_model()
        generator.load_weights(filepath=config.uav_view_generator)

        # tanker
        noise = np.random.uniform(-1, 1, size=[config.size_per_class, config.noise_dim])
        y_tanker = self.gen_labels('tanker', config.size_per_class)
        random_labels = tf.concat(
            [noise, y_tanker], axis=1
        )
        x_tanker = generator.predict(random_labels)

        # container
        noise = np.random.uniform(-1, 1, size=[config.size_per_class, config.noise_dim])
        y_container = self.gen_labels('container', config.size_per_class)
        random_labels = tf.concat(
            [noise, y_container], axis=1
        )
        x_container = generator.predict(random_labels)

        # bulkcarrier
        noise = np.random.uniform(-1, 1, size=[config.size_per_class, config.noise_dim])
        y_bulkcarrier = self.gen_labels('bulkcarrier', config.size_per_class)
        random_labels = tf.concat(
            [noise, y_bulkcarrier], axis=1
        )
        x_bulkcarrier = generator.predict(random_labels)

        # general cargo
        noise = np.random.uniform(-1, 1, size=[config.size_per_class, config.noise_dim])
        y_generalcargo = self.gen_labels('general cargo', config.size_per_class)
        random_labels = tf.concat(
            [noise, y_generalcargo], axis=1
        )
        x_generalcargo = generator.predict(random_labels)

        x_comb = np.concatenate([x_tanker,x_container,x_bulkcarrier,x_generalcargo],axis=0)
        y_comb = np.concatenate([y_tanker,y_container,y_bulkcarrier,y_generalcargo],axis=0)

        return x_comb,y_comb