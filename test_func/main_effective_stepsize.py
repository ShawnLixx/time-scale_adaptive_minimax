import random
from collections import defaultdict
from typing import OrderedDict
from functools import partial
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
 
# import packages from parent directory
sys.path.append('..')
from optimizer.tiada import TiAda, TiAda_Adam, TiAda_wo_max, Adagrad

from tensorboard_logger import Logger

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=8, help='random seed')
parser.add_argument('--n_iter', type=int, default=3000, help='number of gradient calls')
parser.add_argument('--lr_y', type=float, default=0.01, help='learning rate of y')
parser.add_argument('--r', type=float, default=1, help='ratio of stepsize y and stepsize x')
parser.add_argument('--init_x', type=float, default=None, help='init value of x')
parser.add_argument('--init_y', type=float, default=None, help='init value of y')
parser.add_argument('--grad_noise_y', type=float, default=0, help='gradient noise variance')
parser.add_argument('--grad_noise_x', type=float, default=0, help='gradient noise variance')

parser.add_argument('--func', type=str, default='quadratic', help='function name')
parser.add_argument('--L', type=float, default=1, help='parameter for the test function')

parser.add_argument('--optim', type=str, default='adam', help='optimizer')
parser.add_argument('--alpha', type=float, default=0.6, help='parameter for TiAda')
parser.add_argument('--beta', type=float, default=0.4, help='parameter for TiAda')
args = parser.parse_args()

# Set precision to 64
torch.set_default_dtype(torch.float64)

# Different functions
functions = OrderedDict()

L = args.L
functions["quadratic"] = {
        "func":
            lambda x, y: -1/2 * (y ** 2) + L * x * y - (L ** 2 / 2) * (x ** 2),
        }
functions["McCormick"] = {
        "func":
            lambda x, y: torch.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1 + y[0] * x[0] + y[1] * x[1] \
                - 0.5 * (y[0] ** 2 + y[1] ** 2),
        }

optimizers = OrderedDict()
if args.func == 'McCormick':
    # Adam is extremely unstable on McCormick functions, so we need a large eps
    eps = 0.8
    optimizers["Adam"] = partial(torch.optim.Adam, eps=eps)
    optimizers["AMSGrad"] = partial(torch.optim.Adam, amsgrad=True, eps=eps)
else:
    eps = 1e-8
    optimizers["Adam"] = partial(torch.optim.Adam, eps=eps)
    optimizers["AMSGrad"] = partial(torch.optim.Adam, amsgrad=True, eps=eps)

optimizers["AdaGrad"] = Adagrad
optimizers["GDA"] = torch.optim.SGD


# TiAda
optimizers["TiAda"] = TiAda
optimizers["TiAda_Adam"] = TiAda_Adam
optimizers["TiAda_wo_max"] = TiAda_wo_max

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

n_iter = args.n_iter
ratio = args.r

print(f"Function: {args.func}")
print(f"Optimizer: {args.optim}")
fun = functions[args.func]["func"]

if args.func == "McCormick":
    dim = 2
else:
    dim = 1

# Tensorboard
filename = f"./logs/{args.optim}_"
if args.func == 'quadratic':
    filename += f"L{args.L}"
else:
    filename += f"{args.func}"
filename += f"_r_{ratio}_lry_{args.lr_y}"
if 'TiAda' in args.optim:
    filename += f"_a_{args.alpha}_b_{args.beta}"
if args.grad_noise_x != 0:
    filename += f"_noisex_{args.grad_noise_x}"
if args.grad_noise_y != 0:
    filename += f"_noisey_{args.grad_noise_y}"
filename += "_effective_stepsize"

logger = Logger(filename)
logger.config_summary(args)

# learning rate
lr_y = args.lr_y
lr_x = lr_y / ratio

if args.init_x is None:
    init_x = torch.randn(dim)
else:
    init_x = torch.Tensor([args.init_x])
if args.init_y is None:
    init_y = torch.randn(dim)
else:
    init_y = torch.Tensor([args.init_y])
if args.func != 'bilinear':
    print(f"init x: {init_x}, init y: {init_y}")

x = torch.nn.parameter.Parameter(init_x.clone())
y = torch.nn.parameter.Parameter(init_y.clone())

if "NeAda" in args.optim:
    optim_name = args.optim[6:]
else:
    optim_name = args.optim

if args.optim == 'TiAda':
    optim_y = TiAda([y], lr=lr_y, alpha=args.beta, compute_effective_stepsize=True)
    optim_x = TiAda([x], opponent_optim=optim_y, lr=lr_x, alpha=args.alpha,
                    compute_effective_stepsize=True)
elif args.optim == 'TiAda_Adam':
    optim_y = TiAda_Adam([y], lr=lr_y, alpha=args.beta, eps=eps)
    optim_x = TiAda_Adam([x], opponent_optim=optim_y, lr=lr_x,
            alpha=args.alpha, eps=eps)
elif args.optim == 'TiAda_wo_max':
    optim_x = TiAda_wo_max([x], lr=lr_x, alpha=args.alpha)
    optim_y = TiAda_wo_max([y], lr=lr_y, alpha=args.beta)
else:
    optim = optimizers[optim_name]
    optim_x = optim([x], lr=lr_x)
    optim_y = optim([y], lr=lr_y)

i = 0
outer_loop_count = 0
save_gap = 10
while i < n_iter:
    if "NeAda" in args.optim:
        # inner loop
        required_err = 1 / (outer_loop_count + 1)
        inner_step = 0
        inner_err = required_err + 1  # ensure execute at least one step 
        stop_constant = 1 # stop when number of steps >= stop_constant * outer_loop_count
        if args.func == 'quadratic':
            # Stop earlier in quadratic case
            stop_constant = 0.1
        while inner_err > required_err and i < n_iter and inner_step < stop_constant * outer_loop_count:
            inner_step += 1
            # update y
            optim_x.zero_grad()
            optim_y.zero_grad()
            l = -fun(x, y)
            l.backward()
            # stocastic gradient
            y.grad += torch.randn(dim) * args.grad_noise_y
            optim_y.step()

            inner_err = torch.norm(y.grad) ** 2
            i += 1

        if i == n_iter:
            break

        # outer loop
        # update x
        optim_x.zero_grad()
        optim_y.zero_grad()
        l = fun(x, y)
        l.backward()

        # record the deterministic gradient norm
        i += 1
        x_grad_norm = torch.norm(x.grad).item()
        logger.scalar_summary('x_grad', step=i, value=x_grad_norm)
        outer_loop_count += 1
        # stocastic gradient
        x.grad += torch.randn(dim) * args.grad_noise_x
        optim_x.step()

    else:  # other single-loop optimizers
        optim_x.zero_grad()
        optim_y.zero_grad()
        l = fun(x, y)
        l.backward()
        # record gradient first, since we show deterministic gradients norm
        i += 2
        x_grad_norm = torch.norm(x.grad).item()
        if i % save_gap == 0:
            logger.scalar_summary('x_grad', step=i, value=x_grad_norm)
            logger.scalar_summary('x', step=i, value=x.item())
            logger.scalar_summary('y', step=i, value=y.item())
        # stocastic gradient
        y.grad = -y.grad + args.grad_noise_y * torch.randn(dim)
        x.grad += args.grad_noise_x * torch.randn(dim)
        optim_y.step()
        optim_x.step()

        if i % save_gap == 0:
            logger.scalar_summary('x_effective_stepsize', step=i, value=optim_x.effective_stepsize)
            logger.scalar_summary('y_effective_stepsize', step=i, value=optim_y.effective_stepsize)
            logger.scalar_summary('stepsize_ratio_n', step=i, value=optim_x.effective_stepsize / optim_y.effective_stepsize)
    if x_grad_norm > 1e4:
        break
