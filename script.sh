#!/bin/bash

sudo apt-get install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
sudo ln -s /usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3 /lib/ld-linux-armhf.so.3

cd ~/Desktop/
mkdir -p MobileNetv2
cd MobileNetv2

curl -L -o build.zip "https://docs.google.com/uc?export=download&id=1mNK56Sx7JFt7aWQRIJ_cZcSbW6yDHbfQ"
unzip build.zip
cd build

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/arm-linux-gnueabihf/lib/:$(pwd)
chmod +x ./examples/mobilev2
./examples/mobilev2 --data=assets/cnn_data/ --target=neon --image=assets/go_kart_224.ppm --labels=assets/labels.txt

