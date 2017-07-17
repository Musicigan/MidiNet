This repository contains the source code of [MdidNet : A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation](https://arxiv.org/abs/1703.10847)

<img src="network_structure.png" height="75">

## Notes

This model is a slightly modified version of the model that we presented in the above paper, you can find notations in the code if the parameters differ from the paper.

These scripts are refer to [A tensorflow implementation of "Deep Convolutional Generative Adversarial Networks](https://github.com/carpedm20/DCGAN-tensorflow)
Thanks to Taehoon Kim / @carpedm20 for releasing such a decent DCGAN implementaion

## Instructions

The repository contains one trained model, which is  trained under only 50496 midi bars(augmented from 4208 bars), so the generator might sounds not so "creative".

It's quite fun to use Tencorboard to check out the model's training process: "tensorboard --logdir=log/"
You can check out the loss in the training, and the embedding visulizations of real and fake datas.
<img src="embedding.png" height="75">

To train by your own dataset:
1. please follow section4 in the paper for preprocess, EXCEPT!!! The chord arrangement is changed now:
	channel 13 is 0/1 for major/minor, channel 1-12 are the keys. 
2. change line 134-136 to your data path
3. run main.py --is_train True