#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
MyHome=~/DeepLearning/caffe
DATA=$MyHome/examples/leaf
TOOLS=$MyHome/build/tools

TRAIN_DATA_ROOT=$MyHome/examples/leaf/leaf_train_data/
VAL_DATA_ROOT=$MyHome/examples/leaf/leaf_val_data/
EXAMPLE=$DATA
# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=32
  RESIZE_WIDTH=32
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "$EXAMPLE/leaf_train_lmdb"


if [ -d $EXAMPLE/leaf_train_lmdb ]; then
	rm -rf $EXAMPLE/leaf_train_lmdb
	echo "leaf_train_lmdb is removed!"
fi

if [ -d $EXAMPLE/leaf_val_lmdb ]; then
	rm -rf $EXAMPLE/leaf_val_lmdb
	echo "leaf_val_lmdb is removed!"
fi

echo "Creating train lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train_Label.txt \
    $EXAMPLE/leaf_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val_Label.txt \
    $EXAMPLE/leaf_val_lmdb
