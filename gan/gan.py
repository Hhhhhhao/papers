from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataGenerator:
    def __init__(self, batch_size=32, training=True):
        """
        Date generator for training and validation
        :param batch_size: the batch size
        :param training: if True, this data generator is used for training

        return: an iterator with length defined in __len__(self)
        """
        self.batch_size = batch_size
        self.training = training
        # load the dataset
        self.X, self.num = self.load_data()

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.num / float(self.batch_size)))

    def __getitem__(self, idx):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        batch = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch

    def __iter__(self):
        """Create an infinite generator that iterate over the Sequence."""
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item

    def load_data(self):
        """
        load MNIST training data and testing data
        :return: dataset with the shape (n, h, w, 1)
        """
        (X_train, _), (X_test, _) = mnist.load_data()

        if self.training:
            np.random.shuffle(X_train)
            X_train = X_train / 255.
            X_train = np.expand_dims(X_train, axis=-1)
            return X_train, X_train.shape[0]
        else:
            X_test = X_test / 255.
            X_test = np.expand_dims(X_test, axis=-1)
            return X_test, X_test.shape[0]


class GAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        g_optimizer = Adam(0.0001, beta_1=0.5)
        d_optimizer = Adam(0.0001, beta_1=0.5)

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
        x = Dense(256, name='g_dense1')(x_input)
        x = BatchNormalization(name='g_bn1')(x)
        x = LeakyReLU(alpha=0.2, name='g_ac1')(x)
        x = Dense(512, name='g_dense2')(x)
        x = BatchNormalization(name='g_bn2')(x)
        x = LeakyReLU(alpha=0.2, name='g_ac2')(x)
        x = Dense(1024, name='g_dense3')(x)
        x = BatchNormalization(name='g_bn3')(x)
        x = LeakyReLU(alpha=0.2, name='g_ac3')(x)
        x = Dense(np.prod(self.img_shape), activation='sigmoid')(x)
        x_output = Reshape(self.img_shape)(x)

        model = Model(inputs=x_input, outputs=x_output, name='generator')
        model.summary()

        return model

    def build_discriminator(self):

        x_input = Input(shape=self.img_shape,name='d_input0')
        x = Flatten(name='d_flatten0')(x_input)
        x = Dense(1024, name='d_dense1')(x)
        x = BatchNormalization(name='d_bn1')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac1')(x)
        x = Dropout(rate=0.3, name='d_drop1')(x)
        x = Dense(512, name='d_dense2')(x)
        x = BatchNormalization(name='d_bn2')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac2')(x)
        x = Dropout(rate=0.3, name='d_drop2')(x)
        x = Dense(256, name='d_dense3')(x)
        x = BatchNormalization(name='d_bn3')(x)
        x = LeakyReLU(alpha=0.2, name='d_ac3')(x)
        x = Dropout(rate=0.3, name='g_drop3')(x)
        x_output = Dense(1, activation='sigmoid', name="d_dense4")(x)

        model = Model(inputs=x_input, outputs=x_output, name='discriminator')
        model.summary()

        return model

    def train(self, epochs, batch_size=32, k=2, label_smooth=False, sample_intervals=100):
        # initialize losses dict
        losses = {"g_loss":[], "d_loss":[], "d_acc":[], "val_loss":[]}

        # initialize data generators
        val_datagen = DataGenerator(batch_size, training=False)

        # start training
        for epoch in range(epochs):
            print('-' * 15, 'Epoch %d of %f' % (epoch+1, epochs), '-' * 15)
            train_datagen = DataGenerator(batch_size, training=True)
            for t in range(len(train_datagen)):

                # generate batch data
                imgs = train_datagen[t]

                # ----------------------
                #  Train Discriminator
                # ----------------------
                self.discriminator.trainable = True
                d_loss = []
                for i in range(k):
                    # define labels
                    if label_smooth:
                        valid = np.ones((imgs.shape[0], 1)) * 0.9
                    else:
                        valid = np.ones((imgs.shape[0], 1))
                    fake = np.zeros((imgs.shape[0], 1))

                    # generate noise
                    noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))

                    # generate a batch of new images from noise
                    gen_imgs = self.generator.predict(noise)

                    # train the discriminator
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    losses["d_loss"].append(d_loss[0])
                    losses["d_acc"].append(100 * d_loss[1])

                # ----------------------
                #  Train Generator
                # ----------------------
                self.discriminator.trainable = False
                # generate noise and valid labels
                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
                valid = np.ones((imgs.shape[0], 1))
                # train the generator
                # flip the target to resolve the gradient vanish problem
                g_loss = self.combined.train_on_batch(noise, valid)
                losses["g_loss"].append(g_loss)

                # Plot the progress at certain sample intervals
                if t%sample_intervals == 0 or t == len(train_datagen)-1:
                    print("iteration:%d [D loss: %f, acc.: %.2f%%] [G loss: %f] "
                        % (t, d_loss[0], 100 * d_loss[1], g_loss))

            # validation at the end of the epoch
            val_losses = {"g_loss":[], "d_loss":[], "d_acc":[]}
            for t in range(len(val_datagen)):
                # generate validation batch
                imgs = val_datagen[t]

                # validate discriminator and generator
                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                valid = np.ones((imgs.shape[0], 1))
                fake = np.zeros((imgs.shape[0], 1))
                digits = valid + fake
                val_batch = imgs + gen_imgs
                d_loss = self.discriminator.test_on_batch(val_batch, digits)
                g_loss = self.combined.test_on_batch(noise, valid)
                val_losses["d_loss"].append(d_loss[0])
                val_losses["d_acc"].append(d_loss[1])
                val_losses["g_loss"].append(g_loss)
            losses["val_loss"] += val_losses["g_loss"]
            # Plot the progress
            print("validation [D loss: %f, acc.: %.2f%%] [G loss: %f] "
                  % (np.mean(val_losses["d_loss"]), 100 * np.mean(val_losses["d_acc"]), np.mean(val_losses["g_loss"])))
            # save generated samples at epoch end
            self.sample_images(epoch)

        self.generator.save_weights(filepath="./weights/generator.hdf5")
        self.discriminator.save_weights(filepath="./weights/discriminator.hdf5")

        return losses

    def sample_images(self, epoch):
        r, c = 8, 8
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(r, c)
        count = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
            fig.savefig("images/%d.png" % epoch)
            plt.close()


if __name__ == '__main__':
    gan = GAN()
    losses = gan.train(epochs=10, batch_size=64, k=1, label_smooth=True, sample_intervals=100)
    color = ['b', 'g', 'r', 'tab:orange']
    sns.set(color_codes=True)
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
    for key in enumerate(losses.keys()):
        plt.plot(losses[key[1]], color[key[0]])
        plt.xlabel('iterations')

        if key[1] == 'd_acc':
            plt.ylabel('accuracy')
        else:
            plt.ylabel('loss')

        plt.savefig("./" + key[1] + ".png")
        plt.show()
        plt.close()


