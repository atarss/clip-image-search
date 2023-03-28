#!/usr/bin/env python3

import os
import shutil
from glob import glob
from datetime import datetime

import clip
import pymongo
from tqdm import tqdm

import utils
from clip_model import CLIPModel


def import_dir(base_dir, copy=False):
    config = utils.get_config()

    filelist = glob(os.path.join(base_dir, '**/*'), recursive=True)
    filelist = [f for f in filelist if os.path.isfile(f)]

    mongo_client = pymongo.MongoClient("mongodb://{}:{}/".format(config['mongodb-host'], config['mongodb-port']))
    mongo_collection = mongo_client[config['mongodb-database']][config['mongodb-collection']]

    model = CLIPModel(config)
    for filename in tqdm(filelist):
        filetype = utils.get_file_type(filename)
        if filetype is None:
            print("skip file:", filename)
            continue

        image_feature, image_size = model.get_image_feature(filename)
        if image_feature is None:
            print("skip file:", filename)
            continue
        image_feature = image_feature.astype(config['storage-type'])
        
        if copy:
            md5hash = utils.calc_md5(filename)
            new_basename = md5hash + '.' + filetype
            new_full_path = utils.get_full_path(config['import-image-base'], new_basename)

            if os.path.isfile(new_full_path):
                print("duplicate file:", filename)
                continue

            shutil.copy2(filename, new_full_path)
            stat = os.stat(new_full_path)
        else:
            stat = os.stat(filename)
            new_full_path = filename

        image_mtime = datetime.fromtimestamp(stat.st_mtime)
        image_datestr = image_mtime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # save to mongodb
        document = {
            'filename': new_full_path,
            'extension': filetype,
            'height': image_size[1],
            'width': image_size[0],
            'filesize': stat.st_size,
            'date': image_datestr,
            'feature': image_feature.tobytes(),
        }

        x = mongo_collection.insert_one(document)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--copy', action='store_true')
    parser.add_argument('dir')
    args = parser.parse_args()

    import_dir(args.dir, args.copy)


if __name__ == '__main__':
    main()
