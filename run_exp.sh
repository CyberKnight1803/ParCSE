#! /bin/sh


# for p in "cls" "avg"; do
#     for imr in "0.1" "0.15" "0.2" "0.25"; do
#         python train.py \
#         --pooler $p \
#         -i $imr
#     done
# done


for dims in "768" "768 768" "1024 768" "4096 768" "768 1024 768"; do
    python train.py \
    -p "cls" \
    -i "0.2" \
    -d $dims
done