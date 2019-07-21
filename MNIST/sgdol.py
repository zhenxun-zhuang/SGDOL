# Copyright 2019 Zhenxun Zhuang. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================

import os

from torch.optim import Optimizer

class SGDOL(Optimizer):
    """Implement the SGDOL Algorithm.
    
    Description:
    This algorithm was proposed in "Surrogate Losses for Online Learning of 
    Stepsizes in Stochastic Non-Convex Optimization" which can be checked out
    at: https://arxiv.org/abs/1901.09068
    
    The online learning algorithm used here is
    "Follow-The-Regularized-Leader-Proximal" as described in the paper.

    Arguments:
    - params (iterable): iterable of parameters to optimize or dicts
          defining parameter groups.
    - smoothness (float, optional): the assumed smoothness of the loss
          function (default: 10).
    - alpha (float, optional): the parameter alpha used in the inital
          regularizer, a rule of thumb is to set it as smoothness (default: 10)
    """

    def __init__(self, params, smoothness=10.0, alpha=10.0):
        if smoothness < 0.0:
            raise ValueError("Invalid smoothness value: {}".format(smoothness))
        if alpha < 0.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict()
        super(SGDOL, self).__init__(params, defaults)

        self._alpha = alpha
        self._smoothness = smoothness
        
        # Indicate whether we have obtained two stochastic gradients.
        self._is_first_grad = True 
        
        # Initialization.
        self._sum_inner_prods = alpha
        self._sum_grad_normsq = alpha
        self._lr = 1.0 / smoothness

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self._is_first_grad:
            # If it is the first mini-batch, just save the gradient for later
            # use and continue.
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    if p.grad is None:
                        state['first_grad'] = None
                        continue

                    first_grad = p.grad.data
                    if first_grad.is_sparse:
                        raise RuntimeError(
                            'SGDOL does not support sparse gradients')

                    state['first_grad'] = first_grad.clone()
        else:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    second_grad = p.grad.data
                    if second_grad.is_sparse:
                        raise RuntimeError(
                            'SGDOL does not support sparse gradients')

                    state = self.state[p]
                    if state['first_grad'] is None:
                        continue

                    first_grad = state['first_grad']
                    
                    # Accumulate ||g_t||^2_2.
                    first_grad_norm = first_grad.norm()
                    first_grad_normsq = first_grad_norm * first_grad_norm
                    self._sum_grad_normsq += float(first_grad_normsq)
                    
                    # Accumulate <g_t, g'_t>.
                    cip = second_grad.view(-1).dot(first_grad.view(-1))
                    self._sum_inner_prods += float(cip)

            # Compute the step-size of the next round.
            lr = self._lr
            smoothness = self._smoothness
            lr_next = self._sum_inner_prods / (smoothness * self._sum_grad_normsq)
            lr_next = max(min(lr_next, 2.0/smoothness), 0.0)
            self._lr = lr_next

            # Write the step-size to log.
            if not os.path.exists('logs'):
                os.makedirs('logs')

            lr_fname = ''.join(['logs/lr_SGDOL_',
                                '{0}_{1}'.format(
                                    self._smoothness,
                                    self._alpha),
                                '.txt'])
            with open(lr_fname, 'a') as f:
                f.write('{0} {1} {2}\n'.format(lr,
                                               self._sum_inner_prods,
                                               self._sum_grad_normsq))

            # Update the parameters.
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if state['first_grad'] is None:
                        continue

                    first_grad = state['first_grad']

                    p.data.add_(-lr, first_grad)

        self._is_first_grad = not self._is_first_grad

        return loss
