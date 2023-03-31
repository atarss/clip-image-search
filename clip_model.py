#!/usr/bin/env python3
import time
from functools import lru_cache

from PIL import Image
import torch
import clip

import utils


class CLIPModel():
    def __init__(self, config):
        self.config = config
        if 'device' in config:
            self.device = self.config['device']
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model, self.preprocess = self.get_model()
        
    def get_model(self):
        args = {}
        if 'clip-model-download' in self.config:
            args['download_root'] = self.config['clip-model-download']
        return clip.load(self.config['clip-model'], device=self.device, **args)

    def get_image_feature(self, image_path):
        try:
            image = Image.open(image_path)
            image_size = image.size
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        except:
            return None, None  # Failed to load image
        with torch.no_grad():
            feat = self.model.encode_image(image)
            feat = feat.detach().cpu().numpy()
        return feat, image_size

    def get_text_feature(self, text: str):
        text = clip.tokenize([text]).to(self.device)
        feat = self.model.encode_text(text)
        return feat.detach().cpu().numpy()


@lru_cache(maxsize=1)
def get_model() -> CLIPModel:
    config = utils.get_config()
    _time_start = time.time()
    model = CLIPModel(config)
    _time_end = time.time()
    print("[DEBUG] CLIP model loaded in {:.3f} seconds".format(_time_end - _time_start))
    return model


if __name__ == "__main__":
    model = get_model()
    print(model.model)
