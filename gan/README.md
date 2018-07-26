# Generative Adversarial Nets

## Intro

Keras implementation of *Generative Adversarial Networks* with a multilayer perceptron generator and discriminator.

[Code](gan.py)

Paper: [Ian J. Goodfellow et al. *Generative Adversarial Nets*](https://arxiv.org/abs/1406.2661)

## Details

### Results

![results](./images/mnist.gif)


### Discriminator loss

![dloss](./images/d_loss.png)

### Generator loss

![g_loss](./images/g_loss.png)

### Validation loss

![val_loss](./images/val_loss.png)

## Limitations

- As description in the paper, the update of discriminator needs to be synchronized well with the update of generator. Otherwise, the results would tent to collapse. (Generator only recover a small part of real data distribution).
- From the results, the models clearly collapsed (to 7 and 9). One can improve the performance by adding Batch Normalization to both the generator and the discriminator.