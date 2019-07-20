##Usage
For the source folder, you need to first use 'data_prep' to nomalize the data, then choose any algorithm you like to train a predictor.

# Stochastic Gradient Descent with Online Learning
PyTorch implementation of SGDOL from the paper

**[Surrogate Losses for Online Learning of Stepsizes in Stochastic Non-Convex Optimization](https://arxiv.org/abs/1901.09068)**  
Zhenxun Zhuang, Ashok Cutkosky, Francesco Orabona

### Description
Non-convex optimization has attracted lots of attention in recent years, and many algorithms have been developed to tackle this problem. Many of these algorithms are based on the Stochastic Gradient Descent (SGD) proposed by Robbins & Monro over 60 years ago. SGD is intuitive, efficient, and easy to implement. However, it requires a parameter - the stepsize for convergence, which is notoriously tedious and time-consuming to tune. Over the last several years, a plethora of adaptive gradient-based algorithms have emerged to ameliorate this problem. They have proved efficient in reducing the labor of tuning in practice, but many of them lack theoretic guarantees even in the convex setting. In this paper, we propose new surrogate losses to cast the problem of learning the optimal stepsizes for the stochastic optimization of a non-convex smooth objective function onto an online convex optimization problem. This allows the use of no-regret online algorithms to compute optimal stepsizes on the fly. In turn, this results in a SGD algorithm with self-tuned stepsizes that guarantees convergence rates that are automatically adaptive to the level of noise. 

We refer the interested reader to the [paper](https://arxiv.org/abs/1901.09068) for many more details.

### Code & Usage

Here you will find the SGDOL algorithm equipped with the Follow the Regularized Leader online learning algorithm.   

To use this optimization method, you basically do what you should do with AdaGrad, Adam, etc. Specifically, first import it by:
'''
from sgdol import SGDOL
'''

Then tell your model to use SGDOL as the optimizer:
'''
optimizer = SGDOL(net.parameters(), smoothness=10, alpha=1)
'''

Finally, after backward(), simply call:
'''
optimizer.step()
'''
```

For your convenience, I also included a simple example of running a small neural network on the MNIST network.