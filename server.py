#!/usr/bin/env python3
import os
import time
import uuid
from typing import List, Tuple

import numpy as np
import torch
import gradio as gr
import pymongo
from PIL import Image

import utils
from clip_model import CLIPModel
import import_images


def cosine_similarity(query_feature, feature_list):
    print("debug", query_feature.shape, feature_list.shape)
    query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)
    feature_list = feature_list / np.linalg.norm(feature_list, axis=1, keepdims=True)
    sim_score = (query_feature @ feature_list.T)

    return sim_score[0]


def main():
    config = utils.get_config()
    mongo_collection = utils.get_mongo_collection()
    model = CLIPModel(config)

    def search_nearest_image_feature(
            query_feature,
            topn=20,
            minimum_width=0,
            minimum_height=0,
            extension_choice=[],
            mongo_collection=None):

        assert isinstance(minimum_height, int) and isinstance(minimum_width, int)
        assert mongo_collection is not None
        # find all image filename and features
        mongo_query_dict = {}
        if minimum_width > 0:
            mongo_query_dict["width"] = {"$gte": minimum_width}
        if minimum_height > 0:
            mongo_query_dict["height"] = {"$gte": minimum_height}
        if len(extension_choice) > 0:
            mongo_query_dict["extension"] = {"$in": extension_choice}
        
        cursor = mongo_collection.find(mongo_query_dict, {"_id": 0, "filename": 1, "feature": 1})

        filename_list = []
        feature_list = []
        sim_score_list = []
        MAX_SPLIT_SIZE = 8192
        for doc in cursor:
            feature_list.append(np.frombuffer(doc["feature"], config["storage-type"]))
            filename_list.append(doc["filename"])

            if len(feature_list) >= MAX_SPLIT_SIZE:
                feature_list = np.array(feature_list)
                sim_score_list.append(cosine_similarity(query_feature, feature_list))
                feature_list = []

        if len(feature_list) > 0:
            feature_list = np.array(feature_list)
            sim_score_list.append(cosine_similarity(query_feature, feature_list))

        if len(sim_score_list) == 0:
            return [], []

        sim_score = np.concatenate(sim_score_list, axis=0)
        print("[DEBUG] len(sim_score) = {}".format(len(sim_score)))

        top_n_idx = np.argsort(sim_score)[::-1][:topn]
        top_n_filename = [filename_list[idx] for idx in top_n_idx]
        top_n_score = [sim_score[idx] for idx in top_n_idx]

        return top_n_filename, top_n_score


    def submit(prompt:str, topn:int,
               minimum_width:int, minimum_height:int,
               extension_choice) -> List[Tuple[Image.Image, str]]:
        _debug_time_start = time.time()

        with torch.no_grad():
            if isinstance(prompt, str):
                target_feature = model.get_text_feature(prompt)
            else:
                # prompt -> PIL.Image
                image_input = model.preprocess(prompt).unsqueeze(0).to(model.device)
                image_feature = model.model.encode_image(image_input)
                target_feature = image_feature.cpu().detach().numpy()

        topn = int(topn)
        minimum_width = int(minimum_width)
        minimum_height = int(minimum_height)

        filename_list, score_list = search_nearest_image_feature(target_feature, topn=topn,
                                                                 minimum_width=minimum_width,
                                                                 minimum_height=minimum_height, 
                                                                 extension_choice=extension_choice,
                                                                 mongo_collection=mongo_collection)
        ret_list = []
        if len(filename_list) == 0:
            return ret_list

        doc_result = mongo_collection.find({"filename": {"$in": filename_list}}, {"_id": 0, "filename": 1, "width": 1, "height": 1, "filesize": 1, "date": 1})
        doc_result = list(doc_result)
        for filename, score in zip(filename_list, score_list):
            doc = [d for d in doc_result if d["filename"] == filename][0]

            s = ""
            s += "Score = {:.5f}\n".format(score)
            s += (os.path.basename(filename) + "\n")
            s += "{}x{}, filesize={}, {}\n".format(
                doc['width'], doc['height'],
                doc['filesize'], doc['date']
            )

            ret_list.append((filename, s))

        _debug_time_end = time.time()
        print("[DEBUG] total time on submit = {:.6f}s".format(_debug_time_end - _debug_time_start))

        return ret_list


    def upload(image:Image.Image) -> str:
        temp_file_path = "/tmp/" + str(uuid.uuid4()) + ".png"
        image.save(temp_file_path)

        # TODO: resize image to a smaller size if needed
        x = import_images.import_single_image(temp_file_path, mongo_collection, config)
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

        button_prompt.click(submit, inputs=[prompt_textbox, topn, minimum_width, minimun_height, extension_choice], outputs=[gallery])
        button_image.click(submit, inputs=[input_image, topn, minimum_width, minimun_height, extension_choice], outputs=[gallery])
        button_upload.click(upload, inputs=[input_image], outputs=[debug_output])

    demo.launch(server_name=config['server-host'], server_port=config['server-port'])


if __name__ == "__main__":
    main()
