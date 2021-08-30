# -*- coding: utf-8 -*-
"""
This script can be used to download the Images from the urls.txt file one split at a time.
"""

import os
import argparse
import requests
import re

# os.makedirs(save_dir)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='For Getting Inference Arguments')
    parser.add_argument('--urls_path', type=str,
                        help='Path to urls.txt file')
    parser = parser.parse_args()
    args = vars(parser)
    save_dir = os.getcwd()
    save_dir  = os.path.join(save_dir,"/downloaded_images")

    with open(args["urls_path"],"r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()
        # print(line)
        print("full line")
        if "http" in line:
            print("url lisr")

            print(line[3:])
            response = requests.get(line[3:])
            print(response)
            file_path = line[:2]+ ".jpg"
            file_path = os.path.join(save_dir,file_path)
            print(file_path)
            file = open(file_path,"wb")
            file.write(response.content)
            file.close()



