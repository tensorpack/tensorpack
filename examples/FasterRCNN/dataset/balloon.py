import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry

__all__ = ["register_balloon"]


class BalloonDemo(DatasetSplit):
    def __init__(self, base_dir, split):
        assert split in ["train", "val"]
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = os.path.join(base_dir, split)
        assert os.path.isdir(self.imgdir), self.imgdir

    def training_roidbs(self):
        json_file = os.path.join(self.imgdir, "via_region_data.json")
        with open(json_file) as f:
            obj = json.load(f)

        ret = []
        for _, v in obj.items():
            fname = v["filename"]
            fname = os.path.join(self.imgdir, fname)

            roidb = {"file_name": fname}

            annos = v["regions"]

            boxes = []
            segs = []
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = np.stack((px, py), axis=1) + 0.5
                maxxy = poly.max(axis=0)
                minxy = poly.min(axis=0)

                boxes.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])
                segs.append([poly])
            N = len(annos)
            roidb["boxes"] = np.asarray(boxes, dtype=np.float32)
            roidb["segmentation"] = segs
            roidb["class"] = np.ones((N, ), dtype=np.int32)
            roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
            ret.append(roidb)
        return ret


def register_balloon(basedir):
    for split in ["train", "val"]:
        name = "balloon_" + split
        DatasetRegistry.register(name, lambda x=split: BalloonDemo(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "balloon"])


if __name__ == '__main__':
    basedir = '~/data/balloon'
    roidbs = BalloonDemo(basedir, "train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
        imshow(vis)
