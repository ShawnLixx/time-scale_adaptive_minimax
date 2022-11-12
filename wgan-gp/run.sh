for optim in tiada-adam adam
do

for LR in 0.01 0.005 0.001
do

python main.py  \
    --optim $optim \
    --critic_iter 1 \
    --alpha 0.6 \
    --beta 0.4 \
    --lr $LR

done
done
