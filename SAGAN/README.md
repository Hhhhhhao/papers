# Self-Attention GAN

## Introudction

Keras implementation of Self-Attention GANs.


[Code](sagan.py)

Paper: [Ian J. Goodfellow et al. *Self-Attention Generative Adversarial Networks*](https://arxiv.org/abs/1805.08318)


## Update Status

 * [ ] Attention Visualization
 * [ ] Add keras implementation of spectral normlization (a little bit tricky)
 * [ ] Generate Cifar10 and ImageNet images


## Details

### MINIST

#### Training Process

Results gif during training.  


![training](./images/results.gif)

#### Results
Results after training 15 epochs.  


![results](./images/15.png)


### Discriminator loss

Plot of training discriminator fake samples loss.  

![dloss](./images/d_fake_losspng)

Plot of training discriminator real samples loss.  

![dloss](./images/d_real_losspng)

### Discriminator accuracy

Training real samples acc.  


![d_real_acc](./images/d_acc_real.png)

Training fake samples acc.  


![d_real_acc](./images/d_acc_fake.png)


### Generator loss

Plot of training generator loss.  


![g_loss](./images/g_loss.png)

### Validation loss

Plot of validation generator loss.  


![val_loss](./images/val_loss.png)

### Validation accuracy

Validation real samples acc.  


![val_real_acc](./images/val_real_acc.png)

Validation fake samples acc.  


![val_real_acc](./images/val_fake_acc.png)
