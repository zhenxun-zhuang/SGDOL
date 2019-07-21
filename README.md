# Stochastic Gradient Descent with Online Learning
PyTorch implementation of SGDOL from the paper:

**[Surrogate Losses for Online Learning of Stepsizes in Stochastic Non-Convex Optimization](https://arxiv.org/abs/1901.09068)**  
Zhenxun Zhuang, Ashok Cutkosky, Francesco Orabona

### Description
Non-convex optimization has attracted lots of attention in recent years, and many algorithms have been developed to tackle this problem. Many of these algorithms are based on the Stochastic Gradient Descent (SGD) proposed by Robbins & Monro over 60 years ago. SGD is intuitive, efficient, and easy to implement. However, it requires a hand-picked parameter, the stepsize, for (fast) convergence, which is notoriously tedious and time-consuming to tune. Over the last several years, a plethora of adaptive gradient-based algorithms have emerged to ameliorate this problem. They have proved efficient in reducing the labor of tuning in practice, but many of them lack theoretic guarantees even in the convex setting. In this paper, we propose new surrogate losses to cast the problem of learning the optimal stepsizes for the stochastic optimization of a non-convex smooth objective function onto an online convex optimization problem. This allows the use of no-regret online algorithms to compute optimal stepsizes on the fly. In turn, this results in a SGD algorithm with self-tuned stepsizes that guarantees convergence rates that are automatically adaptive to the level of noise. 

We refer the interested reader to the [paper](https://arxiv.org/abs/1901.09068) for many more details.

### Code & Usage

Here you will find the SGDOL algorithm equipped with the Follow-The-Regularized-Leader-Proximal online learning algorithm.   

To use this optimization method, basically do what you normally do with AdaGrad, Adam, etc. Specifically, first import it by:

```
from sgdol import SGDOL
```

Then tell your model to use SGDOL as the optimizer:

```
optimizer = SGDOL(net.parameters(), smoothness=10, alpha=10)
```

Finally, after backward(), simply call:

```
optimizer.step()
```

Note that I choose to record the step-size sequence learned during training, together with the two quantities (the sum of inner products and the sum of gradients norm squared) used to compute the step-size. To disable outputting this information, simply comment the block "Write the step-size to log" (Lines 124-135).

### Example
For your convenience, I also included a simple example of running a small neural network on the MNIST dataset. To run it, simply enter the "MNIST" directory, and use:

```
python main.py
```

Optional arguments:

```
-h, --help            show this help message and exit
--use-cuda            allow the use of CUDA (default: False)
--seed                random seed (default: 0)
--train-epochs        number of epochs to train (default: 30)
--train-batchsize     batchsize in training (default: 100)
--dataroot            location to save the dataset (default: ./data)
--optim-method        choose from {SGDOL,Adam,SGD,Adagrad}, the optimizer to be employed (default: SGDOL)
--smoothness          to be used in SGDOL (default: 10)
--alpha               to be used in SGDOL (default: 10)
--lr                  learning rate of the chosen optimizer (default: 0.001)
```

It will download the MNIST dataset into the specified "dataroot" directory, then use the selected optimizer together with its parameters to train a CNN model. This CNN model consists of two 5*5 convolution layers each of which is activated using ReLU and then maxpooled, followed by two fully connected layers. At the end of each epoch, the last updated model would be evaluated over all training samples to compute the training loss. And after the training is finished, the trained model would be evaluated on the test dataset to compute the test accuracy. All training losses and the test accuracy would be saved to two separate files in the "logs" directory. 
