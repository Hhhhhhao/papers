from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPool2D
from keras.layers import Activation, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


class GAN:
    def __init__(self, dataset='MNIST'):

        if dataset == 'MNIST':
            (self.x_train, _), (self.x_test, _) = mnist.load_data()
            self.x_train = np.expand_dims(self.x_train, axis=-1)
            print(self.x_train.shape)
            self.x_test = np.expand_dims(self.x_test, axis=-1)
            self.ini_size = 7
        elif dataset == 'cifar10':
            (self.x_train, _), (self.x_test, _) = cifar10.load_data()
            self.ini_size = 8
        else:
            raise ValueError("dataset:{} is not valid".format(dataset))

        self.sample_dir = check_folder(dataset + '_samples/')
        self.weights_dir = check_folder(dataset + '_weights/')

        self.img_rows = self.x_train.shape[1]
        self.img_cols = self.x_train.shape[2]
        self.channels = self.x_train.shape[-1]
        print(self.channels)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        g_optimizer = Adam(0.0002, beta_1=0.5)
        d_optimizer = Adam(0.0002, beta_1=0.5)

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])

        # build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generate imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # for the combined model we only train the generator
        self.discriminator.trainable = False

        # the discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # the combined model
        # train generator to fool the discriminator
        self.combined = Model(z, validity, name="generator_and_discriminator")
        self.combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        self.combined.summary()

    def build_generator(self):

        x_input = Input(shape=(self.latent_dim,), name='g_input0')
        x = Dense(self.ini_size * self.ini_size * 128, name='g_dense1')(x_input)
        x = BatchNormalization(name='g_bn1')(x)
        x = Activation('relu', name='g_ac1')(x)

        x = Reshape((self.ini_size, self.ini_size, 128), name='g_reshape1')(x)

        x = UpSampling2D((2, 2), name='g_up2')(x)
        x = Conv2D(128, kernel_size=5, strides=1, padding='same', name='g_conv2')(x)
        x = BatchNormalization(name='g_bn2')(x)
        x = Activation('relu', name='g_ac2')(x)
        x = Conv2D(64, kernel_size=5, strides=1, padding='same', name='g_conv3')(x)
        x = BatchNormalization(name='g_bn3')(x)
        x = Activation('relu', name='g_ac3')(x)
        x = UpSampling2D((2, 2), name='g_up3')(x)
        x = Conv2D(32, kernel_size=5, strides=1, padding='same', name='g_conv4')(x)
        x = BatchNormalization(name='g_bn4')(x)
        x = Activation('relu', name='g_ac4')(x)

        x = Conv2D(3, kernel_size=5, strides=1, padding='same', name='g_conv5')(x)
        x = BatchNormalization(name='g_bn5')(x)
        x = Activation('relu', name='g_ac5')(x)

        x = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', name='g_conv6')(x)
        x_output = Activation('sigmoid', name='g_ac6')(x)

        model = Model(inputs=x_input, outputs=x_output, name='generator')
        model.summary()

        return model

    def build_discriminator(self):

        x_input = Input(shape=self.img_shape,name='d_input0')
        x = Conv2D(32, kernel_size=5, strides=1, padding='same', name='d_conv1')(x_input)
        x = BatchNormalization(name='d_bn1')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac1')(x)
        x = MaxPool2D((2,2))(x)
        x = Conv2D(64, kernel_size=5, strides=1, padding='same', name='d_conv2')(x)
        x = BatchNormalization(name='d_bn2')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac2')(x)
        x = MaxPool2D((2, 2))(x)
        x = Conv2D(128, kernel_size=5, strides=1, padding='same', name='d_conv3')(x)
        x = BatchNormalization(name='d_bn3')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac3')(x)
        x = MaxPool2D((2, 2))(x)
        x = Flatten(name='d_flatten4')(x)
        x = Dense(1024, activation='relu', name="d_dense4")(x)
        x = Dropout(0.2)(x)
        x_output = Dense(1, activation='sigmoid', name="d_dense5")(x)

        model = Model(inputs=x_input, outputs=x_output, name='discriminator')
        model.summary()

        return model

    def train(self, epochs, batch_size=64, label_smooth=False, sample_intervals=100):
        losses = {"g_loss":[], "d_loss":[], "val_loss":[], "d_acc_real":[], "d_acc_fake":[], "val_real_acc":[], "val_fake_acc":[]}

        train_datagen = ImageDataGenerator(
            rotation_range=10,
            rescale=1/255.,
            width_shift_range=0.1,
            height_shift_range=0.1)
        val_datagen = ImageDataGenerator(rescale=1/255.)

        train_datagen.fit(self.x_train)
        val_datagen.fit(self.x_test)

        train_generator = train_datagen.flow(self.x_train, batch_size=batch_size)
        val_generator = val_datagen.flow(self.x_test, batch_size=batch_size)

        self.sample_images(0)
        for epoch in range(epochs):

            print('-' * 15, 'Epoch %d of %f' % (epoch+1, epochs), '-' * 15)
            for t in range(len(train_generator)):
                # generate batch data
                imgs = train_generator[t]

                # ----------------------
                #  Train Discriminator
                # ----------------------
                self.discriminator.trainable = True

                # Adversial ground truths
                if label_smooth:
                    valid = np.ones((imgs.shape[0], 1)) * 0.9
                else:
                    valid = np.ones((imgs.shape[0], 1))

                fake = np.zeros((imgs.shape[0], 1))

                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))

                # generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                losses["d_loss"].append(d_loss[0])
                losses["d_acc_real"].append(100 * d_loss_real[1])
                losses["d_acc_fake"].append(100 * d_loss_fake[1])

                # ----------------------
                #  Train Generator
                # ----------------------
                self.discriminator.trainable = False
                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
                valid = np.ones((imgs.shape[0], 1))
                # train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)
                losses["g_loss"].append(g_loss)
                # Plot the progress
                if t%sample_intervals ==0 or t == len(train_generator)-1:
                    print("iteration:%d [D loss: %a, real acc.: %.2f%%, fake acc.: %.2g%%] [G loss: %g] "
                        % (t, d_loss[0], 100 * d_loss_real[1], 100 * d_loss_fake[1], g_loss))

            val_losses = {"g_loss":[], "d_loss":[], "d_acc_real":[], "d_acc_fake":[]}
            for t in range(len(val_generator)):
                imgs = val_generator[t]
                #print(imgs.shape)
                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                valid = np.ones((imgs.shape[0], 1))
                fake = np.zeros((imgs.shape[0], 1))
                d_loss_real = self.discriminator.test_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.test_on_batch(gen_imgs, fake)
                g_loss = self.combined.test_on_batch(noise, valid)
                d_loss = 0.5 * np.add(d_loss_fake[0], d_loss_real[0])
                val_losses["d_loss"].append(d_loss)
                val_losses["d_acc_real"].append(d_loss_real[1] * 100)
                val_losses["d_acc_fake"].append(d_loss_fake[1] * 100)
                val_losses["g_loss"].append(g_loss)
            losses["val_loss"] += val_losses["g_loss"]
            losses["val_real_acc"] += val_losses["d_acc_real"]
            losses["val_fake_acc"] += val_losses["d_acc_fake"]
            # Plot the progress
            print("validation [D loss: %a, real acc.: %.2f%%, fake acc.:%.2g%%] [G loss: %g] "
                  % (np.mean(val_losses["d_loss"]), np.mean(val_losses["d_acc_real"]), np.mean(val_losses["d_acc_fake"]), np.mean(val_losses["g_loss"])))
            # save generated samples at epoch end
            self.sample_images(epoch+1)

        self.generator.save_weights(filepath=self.weights_dir + "generator.hdf5")
        self.discriminator.save_weights(filepath=self.weights_dir + "discriminator.hdf5")

        return losses

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
            fig.savefig(self.sample_dir+"/%d.png" % epoch)
            plt.close()


if __name__ == '__main__':
    dataset = 'MNIST'
    gan = GAN(dataset)
    loss_dir = check_folder(dataset + '_losses/')
    # set epochs to 30 get good results
    losses = gan.train(epochs=1, batch_size=64, label_smooth=False, sample_intervals=50)

    color = ['b', 'g', 'tab:orange', 'r', 'r', 'r', 'r']
    sns.set(color_codes=True)
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

    for i, key in enumerate(losses.keys()):
        if i <= 2:
            plt.plot(losses[key], color[i])
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.savefig(loss_dir + key[1] + ".png")
            plt.show()
            plt.close()
        elif i == 3 or i == 5:
            plt.plot(losses[key], 'r', label='real samples acc.')
            plt.plot(losses[list(losses.keys())[i+1]], 'b', label='fake samples acc.')
            plt.ylim((0,100))
            plt.xlabel('iterations')
            plt.ylabel('accuracy')
            plt.savefig(loss_dir + key[1] + ".png")
            plt.show()
            plt.close()
        else:
            pass


