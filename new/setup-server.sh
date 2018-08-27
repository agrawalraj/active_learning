#!/usr/bin/env bash

yes| sudo apt-get install libv8-3.14-dev
yes| sudo apt-get install libcurl4-openssl-dev
yes | sudo apt-get install r-base
sudo apt install python3-pip
pip3 install virtualenv
python3 -m virtualenv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
pip3 install --no-cache-dir --upgrade causaldag