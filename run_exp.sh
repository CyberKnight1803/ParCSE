#! /bin/sh

for lr in "3e-5" "3e-4"; do
    for wd in "0" "5e-5" "5e-4"; do
        for p in "cls" "avg_top2" "avg"; do
            python train.py \
            --lr $lr \
            --wd $wd \
            --pooler $p
        done
    done
done