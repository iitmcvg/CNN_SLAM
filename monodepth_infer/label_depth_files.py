'''
Label depth files in a directory

1. Read files

2. Output text file called depth.txt

Format:

# color images
# file: {args.path}
# timestamp filename

1311868164.363181 depth/1311868164.363181.png
'''

import os
import glob
import argparse

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--rgbd_path",default="../r", help="Path to rgbd images")
files = glob.glob("depth/*.png")

with open() as f:

for file in files: