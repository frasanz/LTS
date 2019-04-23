# This is to prepare the data
import os, shutil

# The path to the directry where the original dataset was uncompressed
original_dataset_dir = './kaggle_original_cats_and_dogs'

# The directory where we will store our smaller dataset
base_dir = './cats_and_dogs_small'
os.mkdir(base_dir)
