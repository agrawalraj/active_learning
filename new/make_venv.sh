#!/usr/bin/env bash

pip3 install virtualenv
python3 -m virtualenv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
pip3 install --no-cache-dir --upgrade causaldag

