from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation, Conv2D, UpSampling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
import os


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


class GAN:
    def __init__(self, dataset='MNIST'):
        self.dataset = dataset
        if dataset == 'MNIST':
            (x_train, _), (x_test, _) = mnist.load_data()
            self.x_train = []
            self.x_test = []
            for i, img in enumerate(x_train):
                img = cv2.resize(img, dsize=(32, 32))
                self.x_train.append(img)
            for i, img in enumerate(x_test):
                img = cv2.resize(img, dsize=(32, 32))
                self.x_test.append(img)
            self.x_train = np.expand_dims(self.x_train, axis=-1)
            self.x_test = np.expand_dims(self.x_test, axis=-1)
        elif dataset == 'cifar10':
            (self.x_train, _), (self.x_test, _) = cifar10.load_data()
        else:
            raise ValueError("dataset:{} is not valid".format(dataset))

        self.sample_dir = check_folder(dataset + '_samples/')
        self.weights_dir = check_folder(dataset + '_weights/')

        self.img_rows = self.x_train.shape[1]
        self.img_cols = self.x_train.shape[2]
        self.channels = self.x_train.shape[-1]
        self.layer_num = int(np.log2(self.img_rows)) - 2
        self.ini_size = self.img_rows // (2 ** self.layer_num)
        print(self.ini_size)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        g_optimizer = Adam(0.0002, beta_1=0.5)
        d_optimizer = Adam(0.0004, beta_1=0.5)

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
        ch = 1024
        x_input = Input(shape=(self.latent_dim,))

        x = Dense(self.ini_size * self.ini_size * ch)(x_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((self.ini_size, self.ini_size, ch))(x)

        for i in range(self.layer_num // 2):
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(ch//2, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            ch = ch//2

        for i in range(self.layer_num // 2, self.layer_num):
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(ch//2, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            ch = ch//2

        x = Conv2D(self.channels, kernel_size=3, strides=1, padding='same')(x)
        x_output = Activation('sigmoid')(x)

        model = Model(inputs=x_input, outputs=x_output, name='generator')
        model.summary()

        return model

    def build_discriminator(self):
        ch = 32
        x_input = Input(shape=self.img_shape)
        x = Conv2D(ch, kernel_size=5, strides=1, padding='same')(x_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(self.layer_num // 2):
            x = Conv2D(ch * 2, kernel_size=5, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            ch = ch * 2

        for i in range(self.layer_num // 2, self.layer_num):
            x = Conv2D(ch * 2, kernel_size=5, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            ch = ch * 2

        x = Flatten()(x)
        # x = Dense(1024)(x)
        # X = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        # x = Dropout(0.25)(x)
        x_output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=x_input, outputs=x_output, name='discriminator')
        model.summary()

        return model

    def train(self, epochs, batch_size=64, label_smooth=False, sample_intervals=100):
        losses = {"g_loss":[], "d_real_loss":[], "d_fake_loss":[], "val_loss":[], "d_acc_real":[], "d_acc_fake":[],
                  "val_real_acc":[], "val_fake_acc":[]}

        train_datagen = ImageDataGenerator(
            rescale=1/255.)
        val_datagen = ImageDataGenerator(rescale=1/255.)

        train_datagen.fit(self.x_train)
        val_datagen.fit(self.x_test)

        train_generator = train_datagen.flow(self.x_train, batch_size=batch_size)
        val_generator = val_datagen.flow(self.x_test, batch_size=batch_size)

        self.sample_images(0)
        for epoch in range(epochs):
            d_acc = {"real":[], "fake":[]}
            print('-' * 15, 'Epoch %d of %d' % (epoch+1, epochs), '-' * 15)
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
                #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                losses["d_real_loss"].append(d_loss_real[0])
                losses["d_fake_loss"].append(d_loss_fake[0])
                d_acc["real"].append(100 * d_loss_real[1])
                d_acc["fake"].append(100 * d_loss_fake[1])

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
                    print("iteration:%d [D real loss: %.5a, fake loss: %.5a, real acc.: %.2f%%, fake acc.: %.2f%%] [G "
                          "loss: %.8a] "
                        % (t, d_loss_real[0], d_loss_fake[0], np.mean(d_acc["real"]), np.mean(d_acc["fake"]), g_loss))
                    losses["d_acc_real"].append(np.mean(d_acc["real"]))
                    losses["d_acc_fake"].append(np.mean(d_acc["fake"]))

            val_losses = {"g_loss":[], "d_loss_real":[], "d_loss_fake":[], "d_acc_real":[], "d_acc_fake":[]}
            for t in range(len(val_generator)):
                imgs = val_generator[t]
                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                valid = np.ones((imgs.shape[0], 1))
                fake = np.zeros((imgs.shape[0], 1))
                d_loss_real = self.discriminator.test_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.test_on_batch(gen_imgs, fake)
                g_loss = self.combined.test_on_batch(noise, valid)
                #d_loss = 0.5 * np.add(d_loss_fake[0], d_loss_real[0])
                val_losses["d_loss_real"].append(d_loss_real[0])
                val_losses["d_loss_fake"].append(d_loss_fake[0])
                val_losses["d_acc_real"].append(d_loss_real[1] * 100)
                val_losses["d_acc_fake"].append(d_loss_fake[1] * 100)
                val_losses["g_loss"].append(g_loss)
            losses["val_loss"].append(np.mean(val_losses["g_loss"]))
            losses["val_real_acc"].append(np.mean(val_losses["d_acc_real"]))
            losses["val_fake_acc"].append(np.mean(val_losses["d_acc_fake"]))
            # Plot the progress
            print("validation [D real loss: %.5a, fake loss: %.5a, real acc.: %.2f%%, fake acc.:%.2f%%] [G loss: %.8a] "
                  % (np.mean(val_losses["d_loss_real"]), np.mean(val_losses["d_loss_real"]),
                     np.mean(val_losses["d_acc_real"]), np.mean(val_losses["d_acc_fake"]),
                     np.mean(val_losses["g_loss"])))
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

        if self.dataset == 'MNIST':
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    count += 1
                fig.savefig(self.sample_dir + "/%d.png" % epoch)
                plt.close()
        else:
            for i in range(r):
                for j in range(c):
                    img = gen_imgs[count] * 255
                    img = img.astype(np.uint8)
                    axs[i, j].imshow(img)
                    axs[i, j].axis('off')
                    count += 1
                fig.savefig(self.sample_dir + "/%d.png" % epoch)
                plt.close()


if __name__ == '__main__':
    dataset = 'MNIST'
    gan = GAN(dataset)
    loss_dir = check_folder(dataset + '_losses/')
    # set epochs to 30 get good results
    losses = gan.train(epochs=30, batch_size=64, label_smooth=False, sample_intervals=50)

    color = ['b', 'g', 'g', 'tab:orange', 'r', 'r', 'r', 'r']
    sns.set(color_codes=True)
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

    for i, key in enumerate(losses.keys()):
        if i <= 3:
            plt.plot(losses[key], color[i])
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.savefig(loss_dir + key + ".png")
            plt.show()
            plt.close()
        else:
            plt.plot(losses[key], color[i])
            plt.ylim((0,100))
            plt.xlabel('iterations')
            plt.ylabel('accuracy')
            plt.savefig(loss_dir + key + ".png")
            plt.show()
            plt.close()


