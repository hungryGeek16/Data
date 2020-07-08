#!/bin/bash


sudo apt-get install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

sudo ln -s /usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3 /lib/ld-linux-armhf.so.3

git clone https://github.com/rahulmangalampalli/Data.git

mv Data/build.zip ~/Desktop

cd ~/Desktop

unzip build.zip

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/arm-linux-gnueabihf/lib/:$(pwd)/build

./build/example/v2 --data=build/ --image=build/cnn_data/cat.jpg --labels=build/cnn_data/labels.txt