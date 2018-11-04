from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import os


def write_log(callback, names, logs, batch_no):
    """

    :param callback: keras tensorboard callback
    :param names: loss names
    :param logs: loss values
    :param batch_no: num of iterations
    :return: None
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def create_dir(dir):
    """
    :param dir:directory to create if these directories are not found
    :return exit_code: 0:success -1:failed
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    except Exception as err:
        print('Creating directory error: {0}'.format(err))
        exit(-1)


class GAN:
    def __init__(self):

        self.output_dir = create_dir('./output/')
        self.tensorboard_dir = create_dir('./output/log/')
        self.weights_dir = create_dir('./output/weights/')
        self.samples_dir = create_dir('./output/samples/')

        (self.x_train, _), (self.x_test, _) = mnist.load_data()
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        g_optimizer = Adam(0.001)
        d_optimizer = Adam(0.004)

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])

        # build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generate imgs
        # for the combined model we only train the generator
        self.discriminator.trainable = False
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # the discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # the combined model
        # train generator to fool the discriminator
        self.combined = Model(z, validity, name="generator_and_discriminator")
        self.combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        self.combined.summary()

    def build_generator(self):

        x_input = Input(shape=(self.latent_dim,), name='g_input0')
        x = Dense(256, name='g_dense1')(x_input)
        x = BatchNormalization(name='g_bn1')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(512, name='g_dense2')(x)
        x = BatchNormalization(name='g_bn2')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1024, name='g_dense3')(x)
        x = BatchNormalization(name='g_bn3')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(np.prod(self.img_shape), activation='sigmoid')(x)
        x_output = Reshape(self.img_shape)(x)

        model = Model(inputs=x_input, outputs=x_output, name='generator')
        model.summary()

        return model

    def build_discriminator(self):

        x_input = Input(shape=self.img_shape, name='d_input0')
        x = Flatten(name='d_flatten0')(x_input)
        x = Dense(512, name='d_dense1')(x)
        x = BatchNormalization(name='d_bn2')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac1')(x)
        x = Dense(256, name='d_dense2')(x)
        x = BatchNormalization(name='d_bn2')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac2')(x)
        x = Dense(128, name='d_dense3')(x)
        x = BatchNormalization(name='d_bn3')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac3')(x)
        x_output = Dense(1, activation='sigmoid', name="d_dense4")(x)

        model = Model(inputs=x_input, outputs=x_output, name='discriminator')
        model.summary()

        return model

    def train(self, epochs, batch_size=32):

        # initialize losses names and callback
        tensorboard = TensorBoard(self.tensorboard_dir, batch_size=16, write_graph=True, write_images=True)
        train_generator_names = ['g_bce']
        train_discriminator_names = ['d_bce', 'd_acc']
        valid_names = ['valid_g_bce', 'valid_d_bce', 'valid_d_acc']

        # initialize data generators
        train_datagen = ImageDataGenerator(rescale=1/255.)
        valid_datagen = ImageDataGenerator(rescale=1/255.)
        train_datagen.fit(self.x_train)
        valid_datagen.fit(self.x_test)
        train_generator = train_datagen.flow(self.x_train, batch_size=batch_size)
        valid_generator = valid_datagen.flow(self.x_test, batch_size=batch_size)

        # start training
        total_iteration = 0
        for epoch in range(epochs):
            print('-' * 15, 'Epoch {0} of {1}'.format(epoch+1, epochs), '-' * 15)
            epoch_iteration = 0
            for x_train in train_generator:
                total_iteration += 1
                epoch_iteration += 1
                # ----------------------
                #  Train Discriminator
                # ----------------------
                self.discriminator.trainable = True
                valid = np.ones((x_train.shape[0], 1))
                fake = np.zeros((x_train.shape[0], 1))
                labels = np.concatenate((valid,fake), axis=0)
                # generate noise
                noise = np.random.normal(0, 1, (x_train.shape[0], self.latent_dim))
                # generate a batch of new images from noise
                gen_imgs = self.generator.predict(noise)
                imgs = np.concatenate((x_train, gen_imgs), axis=0)
                # train the discriminator
                d_metrics = self.discriminator.train_on_batch(imgs, labels)
                d_bce = d_metrics[0]
                d_acc = d_metrics[1] * 100
                d_losses = [d_bce, d_acc]
                write_log(tensorboard, train_discriminator_names, d_losses, total_iteration)

                # ----------------------
                #  Train Generator
                # ----------------------
                self.discriminator.trainable = False
                # generate noise and valid labels
                noise = np.random.normal(0, 1, (x_train.shape[0], self.latent_dim))
                valid = np.ones((x_train.shape[0], 1))
                # train the generator
                # flip the target to resolve the gradient vanish problem
                g_bce = self.combined.train_on_batch(noise, valid)
                write_log(tensorboard, train_generator_names, [g_bce], total_iteration)

                # Plot the progress at certain sample intervals
                if epoch_iteration%100 == 0 or epoch_iteration == len(train_generator)-1:
                    print("t:{0}/{1} [D bce_loss:{2:.4f}, acc.:{3:.2f}%] [G bce_loss: {4:.4f}]".format
                          (epoch_iteration, len(train_generator)-1,
                           d_bce, d_acc, g_bce))

            # validation at the end of the epoch
            val_losses = {"g_bce": [], "d_bce": [], "d_acc": []}
            for x_valid in valid_generator:
                # validate discriminator and generator
                noise = np.random.normal(0, 1, (x_valid.shape[0], self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                imgs = np.concatenate((x_valid, gen_imgs), axis=0)
                valid = np.ones((x_valid.shape[0], 1))
                fake = np.zeros((x_valid.shape[0], 1))
                labels = np.concatenate((valid,fake), axis=0)
                d_metrics = self.discriminator.test_on_batch(imgs, labels)
                d_bce = d_metrics[0]
                d_acc = d_metrics[1] * 100
                g_bce = self.combined.test_on_batch(noise, valid)
                val_losses["d_bce"].append(d_bce)
                val_losses["d_acc"].append(d_acc)
                val_losses["g_bce"].append(g_bce)
            valid_losses = [np.mean(val_losses["g_bce"]), np.mean(val_losses["d_bce"]), np.mean(val_losses["d_acc"])]
            write_log(tensorboard, valid_names, valid_losses, epoch+1)
            print("validation epoch:{0}/{1} [D bce_loss:{2:.4f}, acc.:{3:.2f}%] [G bce_loss: {4:.4f}]".format
                  (epoch+1, epochs,
                   valid_losses[0], valid_losses[1], valid_losses[2]))
            # save generated samples at epoch end
            self.sample_images(epoch+1)

        self.generator.save_weights(filepath="./weights/generator.hdf5")
        self.discriminator.save_weights(filepath="./weights/discriminator.hdf5")

    def sample_images(self, epoch):

        r, c = 5, 5
        np.random.seed(1)
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(r, c)
        count = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
            fig.savefig("./output/samples/{0}.png".format(epoch))
            plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100, batch_size=128)


