 #!/bin/bash
 
echo "Script to install g2opy, pypangolin, cmake, cv2"
sudo apt-get install cmake
sudo pip3 install numpy opencv-python PyOpenGL PyOpenGL_accelerate 

git clone https://github.com/uoip/pangolin.git
cd pangolin
mkdir build
cd build
suco cmake ..
sudo make -j8
cd ..
sudo python setup.py install
cd ..

git clone https://github.com/uoip/g2opy.git
cd g2opy
mkdir build
cd build
sudo cmake ..
sudo make -j8
cd ..
sudo python setup.py install


echo "Finished!"