#!/bin/bash

# Exit on errors
set -e

# Create datastore directory for DVC remote storage
mkdir -p datastore

# Initialize DVC
dvc init

# Add localstore as the default DVC remote storage
dvc remote add --default localstore $PWD/datastore

# Pull samples from Git LFS
git lfs pull

# Copy sample1.csv to data/dvc directory
mkdir -p data/dvc

for i in 1 2 3; do
  cp data/processed/twitter-financial-news-sentiment/samples/sample$i.csv data/dvc/sample.csv
  dvc add data/dvc/sample.csv
  dvc push
done


echo "DVC setup complete and samples added."


# git checkout v3.0 data/dvc/sample.csv.dvc