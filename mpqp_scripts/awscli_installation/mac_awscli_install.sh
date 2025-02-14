#!/usr/bin/env bash

curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg ./AWSCLIV2.pkg -target /
echo "AWS CLI V2 installed. Version: "
aws --version