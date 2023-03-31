#!/usr/bin/env python3

import os
import json
import hashlib
from functools import lru_cache

import pymongo
from pymongo.collection import Collection
import clip


@lru_cache(maxsize=1)
def get_config():
    with open('config.json') as json_data_file:
        c = json.load(json_data_file)

    if "clip-model" in c:
        assert c["clip-model"] in clip.available_models()
    if "device" in c:
        assert c["device"] in ["cuda", "cpu"]
    return c


def get_feature_size(model_name):
    if model_name == "ViT-B/32":
        return 512
    elif model_name == "ViT-L/14":
        return 768
    else:
        raise ValueError("Unknown model")  # TODO: complete this table


def get_file_type(image_path):
    libmagic_output = os.popen("file '" + image_path + "'").read().strip()
    libmagic_output = libmagic_output.split(":", 1)[1]
    if "PNG" in libmagic_output:
        return "png"
    if "JPEG" in libmagic_output:
        return "jpg"
    if "GIF" in libmagic_output:
        return "gif"
    if "PC bitmap" in libmagic_output:
        return "bmp"
    return None


@lru_cache(maxsize=1)
def get_mongo_collection() -> Collection:
    config = get_config()
    mongo_client = pymongo.MongoClient("mongodb://{}:{}/".format(config['mongodb-host'], config['mongodb-port']))
    mongo_collection = mongo_client[config['mongodb-database']][config['mongodb-collection']]
    return mongo_collection


def calc_md5(filepath):
    with open(filepath, 'rb') as f:
        md5 = hashlib.md5()
        while True:
            data = f.read(4096)
            if not data:
                break
            md5.update(data)
        return md5.hexdigest()
    

def get_full_path(basedir, basename):
    md5hash, ext = basename.split(".") 
    return "{}/{}/{}/{}".format(basedir, ext, md5hash[:2], basename)
