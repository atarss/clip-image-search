#!/usr/bin/env python3

import torch
import clip

from PIL import Image

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
        if 'clip-model-download-root' in self.config:
            args['download_root'] = self.config['clip-model-download-root']
        return clip.load(self.config['clip-model'], device=self.device)

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


if __name__ == "__main__":
    config = utils.get_config()
    model = CLIPModel(config)
    print(model.model)