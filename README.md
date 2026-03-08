# Deep-learning
This repository holds PyTorch implementations of classic deep learning architectures.

## Architectures

### AlexNet  
[AlexNet (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) is considered to be the first neural network (NN) that performed well on the ImageNet dataset (1.3M training images, 50k validation images and 100k test images containing over 1000 classes). 

The architecture is as follows (image from [Simon J.D. Prince](https://udlbook.github.io/udlbook/)):
<p align="center">
  <img src="assets/ConvAlex.png" alt="AlexNet architecture" width="80%"/>
</p>


For simplicity, I have used the MNIST dataset and adapted the architecture, this is, changing the number of classes from 1000 to 10, and doing some preprocessing on the grayscale images (basically repeating the gray channel 3 times, so the inputs of the original AlexNet would remain the same).


[AlexNet script](Architectures/AlexNet/AlexNet_NN.py)

#### Training

<p align="center">
  <img src="Architectures/AlexNet/Train_test_output_MNIST.png" alt="AlexNet Loss Curve" width="70%"/>
</p>


For the training on the MNIST dataset, we used the following hyperparameters:

| Hyperparameter   | Value        |
|------------------|-------------|
| Optimizer        | Adam        |
| Learning Rate    | 0.001       |
| Loss Function    | CrossEntropyLoss |
| Batch Size       | 32          |
| Epochs           | 20          |
| Dropout          | 0.5         |
| Input Size       | 3 x 224 x 224 |

Both training and test loss decrease rapidly and converge, indicating good learning and no overfitting.

#### Evaluation 

<p align="center">
  <img src="Architectures/AlexNet/Confusion_matrix_MNIST.png" alt="AlexNet Confusion Matrix" width="70%"/>
</p>

As seen in the confusion matrix, most of the classes were correctly identified, yielding a test accuracy of 98.94%.

---

### VGG-19  
[Architectures/VGG-19/VGG19_NN.py](Architectures/VGG-19/VGG19_NN.py)

**Evaluation:**  
<!-- ![VGG19 Loss Curve](results/vgg19_loss_curve.png) -->

*Loss curves and evaluation metrics to be added after training.*

---
