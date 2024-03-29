#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
	echo "Please pass as argument to this script the python binary you want to install mpqp to."
	echo "For example:"
	echo "    $ ./mac-install.sh python3"
	exit 1
fi
python_exec=$1
brew install cmake libomp
if [[ ! -f /usr/local/lib/libomp.dylib ]]; then
	brew_dir=$(which brew | sed 's/\/bin\/brew//g')
	libom_location=$brew_dir/Cellar/libomp
	version=$(ls $libom_location)
	sudo mkdir /usr/local/lib
	sudo ln -sf $libom_location/$version/lib/libomp.dylib /usr/local/lib/libomp.dylib
fi
$python_exec -m pip install mpqp