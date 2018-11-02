 #!/bin/bash
 
echo "Script to install g2opy, pypangolin, cmake, cv2"
sudo apt-get install cmake
sudo pip3 install numpy opencv-python PyOpenGL PyOpenGL_accelerate 
# source activate py36
# cd ~
git clone https://github.com/uoip/pangolin.git
cd pangolin
mkdir build
cd build
sudo cmake ..
sudo make -j8
cd ..
#source activate py36
sudo python3 setup.py install
cd ..

git clone https://github.com/uoip/g2opy.git
cd g2opy
sudo mkdir build
cd build
sudo cmake ..
sudo make -j8
cd ..
# source activate py36
sudo python3 setup.py install


echo "Finished!"
