brew install python@3.9 cmake libomp
echo "alias python=python3.9" >> ~/.zprofile
echo "alias pip=pip3.9" >> ~/.zprofile
source ~/.zprofile
brew_dir=$(which brew | sed 's/\/bin\/brew//g')
libom_location=$brew_dir/Cellar/libomp
version=$(ls $libom_location)
sudo mkdir /usr/local/lib
sudo ln -sf $libom_location/$version/lib/libomp.dylib /usr/local/lib/libomp.dylib
pip install mpqp