#!/bin/bash

set -e

# environment specified at ./data_setup
python3 data_setup/download_data.py
unzip downloaded_data_zips/models.zip
unzip downloaded_data_zips/data.zip
unzip downloaded_data_zips/results.zip
unzip downloaded_data_zips/wheels.zip
