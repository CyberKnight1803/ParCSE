#! /bin/sh


for p in "cls" "avg"; do
    for imr in "0.1" "0.15" "0.2" "0.25"; do
        python train.py \
        --pooler $p \
        -i $imr
    done
done
