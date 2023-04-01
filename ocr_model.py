#!/usr/bin/env python3
import os
from functools import lru_cache
from paddleocr import PaddleOCR

import utils

# experimental code for searching text in images


def download_ocr_model(config):
    download_path = config["ocr-model-download"]
    # check det
    det_download_link_dict = {
        "ch_PP-OCRv3_det_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
    }
    det_model = config["ocr-det-model"]
    if not os.path.exists(os.path.join(download_path, det_model)):
        print("Downloading det model")
        os.system(f"wget {det_download_link_dict[det_model]} -P {download_path}")
        os.system(f"tar -xf {os.path.join(download_path, det_model)}.tar -C {download_path}")
        os.system(f"rm {os.path.join(download_path, det_model)}.tar")
    # check rec
    rec_download_link_dict = {
        "ch_PP-OCRv3_rec_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
    }
    rec_model = config["ocr-rec-model"]
    if not os.path.exists(os.path.join(download_path, rec_model)):
        print("Downloading rec model")
        os.system(f"wget {rec_download_link_dict[rec_model]} -P {download_path}")
        os.system(f"tar -xf {os.path.join(download_path, rec_model)}.tar -C {download_path}")
        os.system(f"rm {os.path.join(download_path, rec_model)}.tar")


class OCRModel:
    def __init__(self, config):
        download_ocr_model(config)

        self.config = config
        self.model = PaddleOCR(
            ocr_version="PP-OCRv3",
            det_model_dir="{}/{}".format(config['ocr-model-download'], config['ocr-det-model']),  # Chinese
            rec_model_dir="{}/{}".format(config['ocr-model-download'], config['ocr-rec-model']),  # Chinese
            use_gpu=(config["device"] == "cuda"),
        )

    def get_ocr_result(self, image_path: str) -> str:
        try:
            ocr_result = self.model.ocr(img=image_path, cls=False)
        except:
            print("Error: ", image_path)
            return None
        ocr_str = ""
        # ocr_full_result = []
        for idx in range(len(ocr_result)):
            for line in ocr_result[idx]:
                ocr_str += (" " + line[1][0])
                # ocr_full_result.append(line)
        ocr_str = ocr_str.strip()
        if len(ocr_str) == 0:
            return None
        return ocr_str  # , ocr_full_result
    

@lru_cache(maxsize=1)
def get_ocr_model():
    config = utils.get_config()
    return OCRModel(config)


if __name__ == "__main__":
    model = get_ocr_model()
    print(model)