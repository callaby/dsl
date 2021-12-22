#!/bin/bash

# step 1 prepare random train and validation sets, validation = 10% from all familias
SUFFIX=$(date +%s)
perl one_step_split_train_test_validation

mkdir data/$SUFFIX
cp -pr data/train.tsv data/$SUFFIX/train.tsv
cp -pr data/test.tsv data/$SUFFIX/test.tsv
cp -pr data/validation.tsv data/$SUFFIX/validation.tsv

# step 2 run the training (small on the hyper-optimized)
python3 train.py

# step 3 get the best for valacc from the train sequential checkpoints
cp -pr $(ls model-valacc-*.h5 | sort | tail -n1) model.h5

echo "----" >> exp

echo $SUFFIX >> exp
echo "----" >> exp
python3 predict.py >> exp
echo "----" >> exp
cp -pr model.h5 models/$SUFFIX.h5
mv history.npy models/$SUFFIX.history.npy

cat exp

rm -f model.h5
rm -f model-valacc-*.h5
