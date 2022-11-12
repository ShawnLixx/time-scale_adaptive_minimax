# Qudratic function

for optim in AdaGrad TiAda NeAda-AdaGrad
do

# Note that the r here is stepsize of y / stepsize of x, while r in the paper
# is stepsize of x / stepsize of y.
for r in 1 2 4 8
do

python main.py --n_iter 8000 \
    --optim $optim \
    --lr_y 0.2 \
    --r $r \
    --init_x 1 \
    --init_y 0.01 \
    --func quadratic \
    --L 2

done
done


# McCormick function stochasitc

for optim in AdaGrad TiAda NeAda-AdaGrad
do

for r in 0.01 0.03 0.05
do

python main.py --n_iter 40000 \
    --optim $optim \
    --lr_y 0.01 \
    --r $r \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \

done
done

# To test the alpha beta in TiAda

python main.py --n_iter 40000 \
    --optim TiAda \
    --lr_y 0.01 \
    --r 0.01 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --alpha 0.5 \
    --beta 0.5


python main.py --n_iter 40000 \
    --optim TiAda \
    --lr_y 0.01 \
    --r 0.01 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --alpha 0.7 \
    --beta 0.3

python main.py --n_iter 40000 \
    --optim TiAda_wo_max \
    --lr_y 0.01 \
    --r 0.01 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --alpha 0.5 \
    --beta 0.5

python main.py --n_iter 40000 \
    --optim TiAda_wo_max \
    --lr_y 0.01 \
    --r 0.01 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --alpha 0.6 \
    --beta 0.4

python main.py --n_iter 40000 \
    --optim TiAda_wo_max \
    --lr_y 0.01 \
    --r 0.01 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --alpha 0.7 \
    --beta 0.3


# Show effective stepsize

for optim in AdaGrad TiAda
do

for r in 0.2
do

python main_effective_stepsize.py --n_iter 8000 \
    --optim $optim \
    --lr_y 0.2 \
    --r $r \
    --init_x 1 \
    --init_y 0.01 \
    --func quadratic \
    --L 2
done
done
