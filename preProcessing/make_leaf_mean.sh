#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

MyHome=~/DeepLearning/caffe
EXAMPLE=$MyHome/examples/leaf
DATA=$MyHome/examples/leaf
TOOLS=$MyHome/build/tools

echo "starting making binaryproto..."

if [ -f $DATA/leaf_mean.binaryproto ]; then
    rm $DATA/leaf_mean.binaryproto
    echo "leaf_mean.binaryproto is removed!"
fi

$TOOLS/compute_image_mean $EXAMPLE/leaf_train_lmdb \
  $DATA/leaf_mean.binaryproto

echo "making binaryproto is Done."
