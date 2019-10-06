# -*- coding: utf-8 -*-

import os
import sys
import tqdm
from contextlib import contextmanager

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit
from tensorpack.utils.fs import download

from sotabencheval.utils import is_server
from sotabencheval.object_detection import COCOEvaluator

# import faster rcnn example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "FasterRCNN"))
from config import finalize_configs, config as cfg  # noqa
from eval import predict_image  # noqa
from dataset import register_coco  # noqa
from dataset.coco import COCODetection  # noqa
from data import get_eval_dataflow  # noqa
from modeling.generalized_rcnn import ResNetFPNModel, ResNetC4Model  # noqa


if is_server():
    DATA_ROOT = "./.data/vision/"
else:  # local settings
    DATA_ROOT = os.path.expanduser("~/data/")
COCO_ROOT = os.path.join(DATA_ROOT, "coco")


register_coco(COCO_ROOT)


@contextmanager
def backup_cfg():
    orig_config = cfg.to_dict()
    yield
    cfg.from_dict(orig_config)


def evaluate_rcnn(model_name, paper_arxiv_id, cfg_list, model_file):
    evaluator = COCOEvaluator(
        root=COCO_ROOT, model_name=model_name, paper_arxiv_id=paper_arxiv_id
    )
    category_id_to_coco_id = {
        v: k for k, v in COCODetection.COCO_id_to_category_id.items()
    }

    cfg.update_args(cfg_list)  # TODO backup/restore config
    finalize_configs(False)
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()
    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit(model_file),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1],
    )
    predictor = OfflinePredictor(predcfg)

    def xyxy_to_xywh(box):
        box[2] -= box[0]
        box[3] -= box[1]
        return box

    df = get_eval_dataflow("coco_val2017")
    df.reset_state()
    for img, img_id in tqdm.tqdm(df, total=len(df)):
        results = predict_image(img, predictor)
        res = [
            {
                "image_id": img_id,
                "category_id": category_id_to_coco_id.get(
                    int(r.class_id), int(r.class_id)
                ),
                "bbox": xyxy_to_xywh([round(float(x), 4) for x in r.box]),
                "score": round(float(r.score), 3),
            }
            for r in results
        ]
        evaluator.add(res)
        if evaluator.cache_exists:
            break

    evaluator.save()


download(
    "http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN2x.npz",
    "./",
    expect_size=165362754)
with backup_cfg():
    evaluate_rcnn(
        "Mask R-CNN (ResNet-50-FPN, 2x)", "1703.06870", [],
        "COCO-MaskRCNN-R50FPN2x.npz",
    )


download(
    "http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN2xGN.npz",
    "./",
    expect_size=167363872)
with backup_cfg():
    evaluate_rcnn(
        "Mask R-CNN (ResNet-50-FPN, GroupNorm)", "1803.08494",
        """FPN.NORM=GN BACKBONE.NORM=GN
FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head
FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head""".split(),
        "COCO-MaskRCNN-R50FPN2xGN.npz",
    )


download(
    "http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz",
    "./",
    expect_size=355680386)
with backup_cfg():
    evaluate_rcnn(
        "Mask R-CNN (ResNet-101-FPN, GN, Cascade)", "1811.08883",
        """
    FPN.CASCADE=True BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3] FPN.NORM=GN
    BACKBONE.NORM=GN FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head
    FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head""".split(),
        "COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz",
    )
