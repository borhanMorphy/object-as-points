from typing import Optional, Callable, Dict, Tuple, List
from collections import defaultdict

import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import CocoDetection


def default_collate_fn(samples):
    fetched_data = defaultdict(list)
    for sample in samples:
        for key, val in sample.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            elif key == "target":
                val = (
                    torch.from_numpy(val[0]),
                    torch.from_numpy(val[1]),
                    torch.from_numpy(val[2]),
                )

            fetched_data[key].append(val)

    fetched_data["image"] = torch.stack(fetched_data["image"], dim=0).float().permute(0, 3, 1, 2)
    if "target" in fetched_data:
        cls_targets, offset_targets, shape_targets = zip(*fetched_data["target"])
        fetched_data["target"] = (
            torch.stack(cls_targets, dim=0),
            torch.stack(offset_targets, dim=0),
            torch.stack(shape_targets, dim=0),
        )

    return fetched_data

class Coco(CocoDetection):

    def __init__(self, root: str, annFile: str, transforms: Optional[Callable] = None, target_generator = None) -> None:
        super().__init__(root, annFile)

        self._transforms = transforms
        self._obj_id_mappings = {
            i: self.coco.cats[cat_id]["name"] for i, cat_id in enumerate(self.coco.cats.keys())
        }
        self._rev_obj_id_mappings = {
            cat_name: i for i, cat_name in self._obj_id_mappings.items()
        }
        self._target_generator = target_generator

    @property
    def target_generator(self):
        return self._target_generator

    @target_generator.setter
    def target_generator(self, target_generator):
        self._target_generator = target_generator

    @property
    def num_classes(self) -> int:
        return len(self._obj_id_mappings.keys())

    @property
    def labels(self) -> List[str]:
        return list(sorted(self._rev_obj_id_mappings.keys()))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict]:
        image, target = super().__getitem__(index)

        image = np.array(image, dtype=np.uint8)

        bboxes = list()
        labels = list()

        for t in target:
            x, y, w, h = t["bbox"]
            if min(w, h) <= 0:
                # skip target if width or height is 0
                continue
            bboxes.append(
                [float(x), float(y), float(x + w), float(y + h)]
            )
            labels.append(
                self.coco.cats[t["category_id"]]["name"]
            )

        data = dict(
            image=image,
            bboxes=bboxes,
            labels=labels
        )

        if self._transforms:
            data = self._transforms(**data)

        data["label_ids"] = [self.label2id(label) for label in data["labels"]]

        if self._target_generator:
            data["target"] = self._target_generator.build_targets(
                *data["image"].shape[:2],
                np.array(data["bboxes"], dtype=np.float32),
                np.array(data["label_ids"], dtype=np.int32),
            )
        return data

    def id2label(self, idx: int) -> str:
        return self._obj_id_mappings[idx]

    def label2id(self, label: str) -> int:
        return self._rev_obj_id_mappings[label]

    def get_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn = default_collate_fn,
        shuffle: bool = False,
        **kwargs,
    ) -> DataLoader:
        torch.utils

        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
            **kwargs,
        )

