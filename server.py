#!/usr/bin/env python3
import os
import time
import uuid
from typing import List, Tuple, Any

import numpy as np
import torch
import gradio as gr
import pymongo
from PIL import Image

import utils
from clip_model import get_model
import import_images


def cosine_similarity(query_feature, feature_list):
    print("debug", query_feature.shape, feature_list.shape)
    query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)
    feature_list = feature_list / np.linalg.norm(feature_list, axis=1, keepdims=True)
    sim_score = (query_feature @ feature_list.T)

    return sim_score[0]


class SearchServer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.feat_dim = utils.get_feature_size(config['clip-model'])

        self.model = get_model()
        self.mongo_collection = utils.get_mongo_collection()
        self._MAX_SPLIT_SIZE = 8192

    def _get_search_filter(self, args):
        ret = {}
        if len(args) == 0: return ret
        if 'minimum_width' in args:
            ret['width'] = {'$gte': int(args['minimum_width'])}
        if 'minimum_height' in args:
            ret['height'] = {'$gte': int(args['minimum_height'])}
        if 'extension_choice' in args and len(args['extension_choice']) > 0:
            ret['extension'] = {'$in': args['extension_choice']}
        return ret

    def search_nearest_clip_feature(self, query_feature, topn=20, search_filter_options={}):
        mongo_query_dict = self._get_search_filter(search_filter_options)
        cursor = self.mongo_collection.find(mongo_query_dict, {"_id": 0, "filename": 1, "feature": 1})

        filename_list = []
        feature_list = []
        sim_score_list = []
        for doc in cursor:
            feature_list.append(np.frombuffer(doc["feature"], self.config["storage-type"]))
            filename_list.append(doc["filename"])

            if len(feature_list) >= self._MAX_SPLIT_SIZE:
                feature_list = np.array(feature_list)
                sim_score_list.append(cosine_similarity(query_feature, feature_list))
                feature_list = []

        if len(feature_list) > 0:
            feature_list = np.array(feature_list)
            sim_score_list.append(cosine_similarity(query_feature, feature_list))

        if len(sim_score_list) == 0:
            return [], []

        sim_score = np.concatenate(sim_score_list, axis=0)

        top_n_idx = np.argsort(sim_score)[::-1][:topn]
        top_n_filename = [filename_list[idx] for idx in top_n_idx]
        top_n_score = [float(sim_score[idx]) for idx in top_n_idx]

        return top_n_filename, top_n_score

    def convert_result_to_gradio(self, filename_list: List[str], score_list: List[float]):
        doc_result = self.mongo_collection.find(
            {"filename": {"$in": filename_list}},
            {"_id": 0, "filename": 1, "width": 1, "height": 1, "filesize": 1, "date": 1})
        doc_result = list(doc_result)
        filename_to_doc_dict = {d['filename']: d for d in doc_result}
        ret_list = []
        for filename, score in zip(filename_list, score_list):
            doc = filename_to_doc_dict[filename]

            s = ""
            s += "Score = {:.5f}\n".format(score)
            s += (os.path.basename(filename) + "\n")
            s += "{}x{}, filesize={}, {}\n".format(
                doc['width'], doc['height'],
                doc['filesize'], doc['date']
            )

            ret_list.append((filename, s))
        return ret_list

    def serve(self):
        server = self
        def _gradio_search_image(query, topn, minimum_width, minimum_height, extension_choice):
            with torch.no_grad():
                if isinstance(query, str):
                    target_feature = server.model.get_text_feature(query)
                elif isinstance(query, Image.Image):
                    image_input = server.model.preprocess(query).unsqueeze(0).to(server.model.device)
                    image_feature = server.model.model.encode_image(image_input)
                    target_feature = image_feature.cpu().detach().numpy()
                else:
                    assert False, "Invalid query (input) type"

            search_options = {
                "minimum_width": minimum_width,
                "minimum_height": minimum_height,
                "extension_choice": extension_choice,
            }

            filename_list, score_list = server.search_nearest_clip_feature(
                target_feature, topn=int(topn), search_filter_options=search_options)
            
            return server.convert_result_to_gradio(filename_list, score_list)

        def _gradio_upload(image:Image.Image) -> str:
            temp_file_path = "/tmp/" + str(uuid.uuid4()) + ".png"
            image.save(temp_file_path)

            # TODO: resize image to a smaller size if needed
            x = import_images.import_single_image(temp_file_path, server.mongo_collection, server.config)
            os.remove(temp_file_path)
            if x is None:
                return "file not uploaded"
            else:
                return str(x)

        # build gradio app
        with gr.Blocks() as demo:
            heading = gr.Markdown("# CLIP Image Search Demo")
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_textbox = gr.Textbox(lines=8, label="Prompt")
                    button_prompt = gr.Button("Search Text").style(size="lg")
                with gr.Column(scale=2):
                    input_image = gr.Image(label="Image", type="pil")
                    with gr.Row():
                        button_image = gr.Button("Search Image").style(size="lg")
                        button_upload = gr.Button("Upload Image").style(size="lg")

            with gr.Accordion("Search options", open=False):
                extension_choice = gr.CheckboxGroup(["jpg", "png", "gif"], label="extension", info="choose extension for search")
                with gr.Row():
                    topn = gr.Number(value=16, label="topn")
                    minimum_width = gr.Number(value=0, label="minimum_width")
                    minimun_height = gr.Number(value=0, label="minimum_height")
            with gr.Accordion("Debug output", open=False):
                debug_output = gr.Textbox(lines=1)

            gallery = gr.Gallery(label="results").style(grid=4, height=6)

            button_prompt.click(_gradio_search_image, inputs=[prompt_textbox, topn, minimum_width, minimun_height, extension_choice], outputs=[gallery])
            button_image.click(_gradio_search_image, inputs=[input_image, topn, minimum_width, minimun_height, extension_choice], outputs=[gallery])
            button_upload.click(_gradio_upload, inputs=[input_image], outputs=[debug_output])

        demo.launch(server_name=config['server-host'], server_port=config['server-port'])


if __name__ == "__main__":
    config = utils.get_config()
    server = SearchServer(config)
    
    server.serve()