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
parser.add_argument("--rgbd_path",default="../rgbd_dataset_freiburg2_desk_OURS", help="Path to rgbd images")
parser.add_argument("--depth_label", default="depth", help="Depth label to use, useful for different nets")
parser.add_argument("--output",default="depth.txt", help="Output file to produce")
args= parser.parse_args()
files = glob.glob("{}/rgb/*.png".format(os.path.join(args.rgbd_path)))

print(files)

dataset = args.rgbd_path.split("/")[-1]
output_path = os.path.join(args.rgbd_path,args.output)

with open(output_path,"w") as f:
    f.write("# depth maps \n# file: '{}'\n# timestamp filename\n".format(dataset))

with open(output_path,"a") as f:
    for file in files:
    
        file_path = file.split("/")[-1]
        timestamp = file_path.split(".p")[0]
        f.write("{} {}/{}\n".format(timestamp, args.depth_label,file_path))