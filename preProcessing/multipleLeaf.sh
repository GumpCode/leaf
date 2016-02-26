#!/usr/bin/env sh
#processing the data of training

HomeDir=~/DeepLearning/caffe/examples/mutipleLeaf
TOOLS=~/DeepLearning/caffe/build/tools
DataDir=~/DeepLearning/DataProcessing/mixSamples
TrainingDir=$HomeDir/leaf_train_data
ValuatingDir=$HomeDir/leaf_val_data
LabelOfPos=$HomeDir/train_Label.txt
LabelOfNeg=$HomeDir/val_Label.txt
trainingRate=0.66

echo "Starting Pre-processing the training data..."

if [ -d $HomeDir/leaf_train_data ]; then
    rm -rf $HomeDir/leaf_train_data
    echo "Old leaf_train_data is removed!"
fi

if [ -d $HomeDir/leaf_val_data ]; then
    rm -rf $HomeDir/leaf_val_data
    echo "Old leaf_val_data is removed!"
fi

python DataProcessing.py $DataDir \
        $TrainingDir $ValuatingDir  \
        $LabelOfPos $LabelOfNeg  \
        $trainingRate 
echo "Pre-processing is finished!"

echo "Starting creating the input data..."
sh $HomeDir/create_leaf.sh
echo "Creating data is finished!"

echo "Starting making binaryproto..."
if [ -f $HomeDir/leaf_mean.binaryproto ]; then
    rm $HomeDir/leaf_mean.binaryproto
    echo "old leaf_mean.binaryproto is removed!"
fi

$TOOLS/compute_image_mean $HomeDir/leaf_train_lmdb \
    $HomeDir/leaf_mean.binaryproto
echo "making binaryproto is finished!"

echo "Starting making data of npy..."
if [ -f $HomeDir/leaf_mean.npy ]; then
    rm $HomeDir/leaf_mean.npy
fi

python $HomeDir/convert_protomean.py $HomeDir/leaf_mean.binaryproto \
           $HomeDir/leaf_mean.npy

if [ -f $HomeDir/leaf_mean.npy ]; then
    echo "leaf_mean.npy is finished!"
fi

echo "All completed!"
