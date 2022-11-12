import os
from os.path import join
import random

import numpy as np
import tensorflow as tf
import torch

from utils.model import get_model
from utils.dataset import get_dataset
from utils.optim import get_optim
from utils.train import train_step
# from utils.train import generator_max_eigen, critic_max_eigen
from utils.tool import cycle

from config import args

TEMP_DIR = 'temp_store'

if __name__ == "__main__":
    # code reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    batchsize = args.batchsize
    train_loader, test_loader = get_dataset()
    batches_in_epoch = len(train_loader)
    
    # Use GPU is available else use CPU.
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")
    args.device = device

    net = get_model()

    # Create optimizers.
    optim_x = get_optim(net.parameters(), x=True)

    # Tensorboard
    if args.model_name is None:
        model_name = f'{args.dataset}_{args.optim}_{args.lr_x}_{args.lr_y}_eps_{args.epsilon}_g_{args.gamma}_i_{args.n_inner}'
        if 'tiada' in args.optim:
            model_name += f'_tiada_{args.alpha}_{args.beta}'
        model_name += f"_seed_{args.seed}"
    else:
        model_name = args.model_name
    log_dir = join('logs', model_name)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Write config
    with summary_writer.as_default():
        tf.summary.text("config",
                [[k, str(w)] for k, w in sorted(vars(args).items())],
                step=0)
    
    # Training loop
    num_epoch = args.num_epoch
    total_steps = num_epoch * batches_in_epoch
    args.total_steps = total_steps
    args.step = 0
    args.outer_step = 0
    data_iter = iter(cycle(train_loader))
    # record a image for a total of num_recorded times
    num_record = num_epoch
    num_recorded = 0
    record_gap = total_steps // num_record

    total_grad_norm_y = torch.zeros(1).to(args.device)

    while args.step < total_steps:

        net.train()

        results = train_step(data_iter, net, optim_x, total_grad_norm_y)
        args.outer_step += 1

        # Write loss
        with summary_writer.as_default():
            tf.summary.scalar('x grad norm', results["x_grad_norm"],
                    step=args.step)
            tf.summary.scalar('y grad norm', results["y_grad_norm"],
                    step=args.step)

            tf.summary.scalar('classification loss', results["classification_loss"],
                    step=args.step)
            tf.summary.scalar('total loss', results["total_loss"],
                    step=args.step)
            tf.summary.scalar('total grad norm y', results["y_total_grad_sum"],
                    step=args.step)


        # Exist if nan
        if np.isnan(results["classification_loss"]):
            with summary_writer.as_default():
                tf.summary.text("nan", "nan", step=args.step)
            exit(0)
