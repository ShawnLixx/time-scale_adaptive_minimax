# Comparison of AdaGrad-like

for optim in adagrad
do
for ninner in 1 15
do

for LR_X in 0.1 0.05 0.01 0.005
do
for LR_Y in 0.1 0.05 0.01 0.005
do

python main.py  \
        --lr_x  $LR_X \
        --lr_y  $LR_Y \
        --num_epoch 200 \
        --optim "$optim" \
        --n_inner "$ninner" \
        --dataset mnist

done
done
done
done

for optim in tiada neada-adagrad
do

for LR_X in 0.1 0.05 0.01 0.005
do
for LR_Y in 0.1 0.05 0.01 0.005
do

python main.py  \
        --lr_x  $LR_X \
        --lr_y  $LR_Y \
        --num_epoch 200 \
        --optim "$optim" \
        --n_inner 1 \
        --dataset mnist

done
done
done


# Comparison of Adam-like

for optim in adam
do
for ninner in 1 15
do

for LR_X in 0.001 0.0005 0.0001
do
for LR_Y in 0.1 0.05 0.005 0.001
do

python main.py  \
        --lr_x  $LR_X \
        --lr_y  $LR_Y \
        --num_epoch 200 \
        --optim "$optim" \
        --n_inner "$ninner" \
        --dataset mnist

done
done
done
done

for optim in tiada-adam neada-adam
do

for LR_X in 0.001 0.0005 0.0001
do
for LR_Y in 0.1 0.05 0.005 0.001
do

python main.py  \
        --lr_x  $LR_X \
        --lr_y  $LR_Y \
        --num_epoch 200 \
        --optim "$optim" \
        --n_inner 1 \
        --dataset mnist

done
done
done
