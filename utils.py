#!/usr/bin/env python3

import os
import json
import hashlib

import clip

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
