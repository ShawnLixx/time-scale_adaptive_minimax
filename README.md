# TiAda: A Time-scale Adaptive Algorithm for Nonconvex Minimax Optimization

Code for paper ["TiAda: A Time-scale Adaptive Algorithm for Nonconvex Minimax Optimization"](https://arxiv.org/abs/2210.17478).
Xiang Li, Junchi Yang, Niao He.

The code for the three tasks mentioned in the paper is store in sub-directories with corresponding names.

The following packages of Python are required for running the code:
````
torch
torchvision
matplotlib
numpy
tensorflow
tensorboard
Pillow
scikit_learn
scipy
six
````

For each task, after entering the folder, simply use
````
bash run.sh
````
to run the experiments described in the paper.

To visualize the results, use tensorboard as
````
tensorboard --logdir logs
````


we used code from https://github.com/Louis-udm/Reproducing-certifiable-distributional-robustness for distributional robustness optimization experiments, and https://github.com/Zeleni9/pytorch-wgan for WGAN-GP experiments.
