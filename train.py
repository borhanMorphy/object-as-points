from typing import Dict, List, Tuple
from collections import defaultdict
import argparse
import json
import os

from tqdm import tqdm
from pydantic import BaseModel

import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.lite import LightningLite

import centernet

class TrainerConfig(BaseModel):
    batch_size: int = 16
    target_batch_size: int = 128
    max_epochs: int = 140
    learning_rate: float = 5e-4

    num_workers: int = 0
    target_image_size: int = 512

    accelerator: str = "gpu"
    precision: int = 32

    validate_every_n_epoch: int = 1

    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    max_pixel_value: float = 1.0


class LiteTrainer(LightningLite):
    def run(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
            batch_size: int = None,
            num_epochs: int = 1,
            validate_every_n_epoch: int = 1,
            resume: str = None,
            learning_rate: float = 5e-4,
            save_path: str = "checkpoints"):

        template_model_name = "epoch_{epoch}_map_{metric_val}_centernet2d_resnet18dcn.ckpt"
        start_epoch = 0
        best_metric_val = -1

        train_dl, val_dl = self.setup_dataloaders(train_dl, val_dl)

        optimizer, scheduler = model.configure_optimizers(learning_rate=learning_rate)

        model, optimizer = self.setup(model, optimizer)

        if resume:
            _, start_epoch, _, best_metric_val, *_ = os.path.basename(resume).split("_")
            start_epoch = int(start_epoch)
            best_metric_val = int(best_metric_val) / 100
            print("loading model from {}".format(resume))
            model.load_state_dict(torch.load(resume, map_location=self.device))

            # update scheduler
            scheduler.last_epoch = start_epoch - 1
            # update 
            scheduler._step_count = start_epoch

        print(scheduler.state_dict())

        if batch_size is None:
            batch_size = train_dl.batch_size

        accumulated_grad_batches =  max(batch_size // train_dl.batch_size, 1)

        os.makedirs(save_path, exist_ok=True)

        for epoch in range(start_epoch, num_epochs):
            print("running epoch [{}/{}]".format(epoch + 1, num_epochs))

            print("running training loop")
            self.run_training_loop(model, optimizer, train_dl, accumulated_grad_batches=accumulated_grad_batches)

            scheduler.step()

            if (epoch + 1) % validate_every_n_epoch != 0:
                continue

            print("running validation loop")
            metrics = self.run_validation_loop(model, val_dl)

            for metric, value in metrics.items():
                print(f"\t{metric} -> {value:.3f}")

            # TODO move to logging
            with open(os.path.join(save_path, f"metrics_epoch_{epoch+1}.json"), "w") as foo:
                json.dump(metrics, foo)

            if metrics["map"] > best_metric_val:
                print("found better value {} -> {}".format(best_metric_val, metrics["map"]))
                best_metric_val = metrics["map"]
                model_save_name = template_model_name.format(epoch=epoch+1, metric_val=int(best_metric_val*100))
                self.save(
                    model.state_dict(),
                    os.path.join(save_path, model_save_name),
                )

    def run_training_loop(self, model: nn.Module, optimizer, dl: DataLoader, accumulated_grad_batches: int = 1):
        model.train()
        optimizer.zero_grad()
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
            targets = batch["target"]

            # compute logits
            logits = model.forward(
                batch["image"]
            )

            # compute loss
            loss = model.module.compute_loss(logits, targets)
            # loss: dict of losses

            self.backward(loss["loss"] / accumulated_grad_batches)

            # TODO log loss in logging

            if (batch_idx + 1) % accumulated_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()


    def run_validation_loop(self, model: nn.Module, dl: DataLoader) -> Dict:
        model.eval()
        mean_metrics = defaultdict(torchmetrics.MeanMetric)
        map_metrics = MeanAveragePrecision(box_format="xyxy")

        for batch in tqdm(dl):
            batch_size = batch["image"].shape[0]

            targets = batch["target"]

            # compute logits
            with torch.no_grad():
                logits = model.forward(
                    batch["image"]
                )

            # compute loss
            loss = model.module.compute_loss(logits, targets)
            # loss: dict of losses
            for key, val in loss.items():
                mean_metrics[key].update(val.cpu())

            metric = model.module.compute_metrics(logits, targets)
            # metric: dict of metrics

            preds = model.module.head.decode(*logits, score_threshold=0.2, keep_n=50)
            # preds: N,7

            batch_preds: List[Dict[str, Tensor]] = list()
            batch_gts: List[Dict[str, Tensor]] = list()

            for batch_idx in range(batch_size):
                mask = preds[:, 0] == batch_idx
                batch_preds.append({
                    "boxes": preds[mask, 1:5].cpu(),
                    "scores": preds[mask, 5].cpu(),
                    "labels": preds[mask, 6].cpu(),
                })
                batch_gts.append({
                    "boxes": torch.tensor(batch["bboxes"][batch_idx]),
                    "labels": torch.tensor(batch["label_ids"][batch_idx]),
                })

            map_metrics.update(batch_preds, batch_gts)

            for key, val in metric.items():
                mean_metrics[key].update(val)

        metrics = {
            key: metric.compute().item()
            for key, metric in mean_metrics.items()
        }

        for key, val in map_metrics.compute().items():
            metrics[key] = val.item()

        return metrics


def main(args):

    training_config = TrainerConfig(
        batch_size=args.batch_size,
        target_batch_size=args.target_batch_size,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        target_image_size=args.target_size,
        precision=args.precision,
        validate_every_n_epoch=args.validate_every_n_epoch,
    )

    train_ds = centernet.dataset.Coco(
        "./data/coco/train2017",
        "./data/coco/annotations/instances_train2017.json",
    )
    val_ds = centernet.dataset.Coco(
        "./data/coco/val2017",
        "./data/coco/annotations/instances_val2017.json",
    )
    model = centernet.module.CenterNet2D(train_ds.num_classes)

    train_ds._transforms = model.train_transforms(
        target_size=training_config.target_image_size,
        min_area=2**2,
        mean=training_config.mean,
        std=training_config.std,
        max_pixel_value=training_config.max_pixel_value,
    )
    train_ds.target_generator = centernet.target.detection_2d.Detection2DTarget(
        output_stride=4,
        num_classes=train_ds.num_classes,
    )
    train_dl = train_ds.get_dataloader(
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        shuffle=True,
    )

    val_ds._transforms = model.transforms(
        target_size=training_config.target_image_size,
        min_area=2**2,
        mean=training_config.mean,
        std=training_config.std,
        max_pixel_value=training_config.max_pixel_value,
    )
    val_ds.target_generator = centernet.target.detection_2d.Detection2DTarget(
        output_stride=4,
        num_classes=val_ds.num_classes,
    )
    val_dl = val_ds.get_dataloader(
        batch_size=training_config.batch_size,
        num_workers=0,
    )

    LiteTrainer(accelerator=training_config.accelerator, precision=training_config.precision).run(
        model,
        train_dl,
        val_dl,
        batch_size=training_config.target_batch_size,
        num_epochs=training_config.max_epochs,
        learning_rate=training_config.learning_rate,
        validate_every_n_epoch=training_config.validate_every_n_epoch,
        resume=args.resume,
    )

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", "-bs", type=int, default=16)
    ap.add_argument("--target-batch-size", "-tbs", type=int, default=128)
    ap.add_argument("--epochs", "-e", type=int, default=140)
    ap.add_argument("--learning-rate", "-lr", type=float, default=5e-4)
    ap.add_argument("--num-workers", "-n", type=int, default=0)
    ap.add_argument("--target-size", "-s", type=int, choices=[2**i for i in range(7, 10)], default=512)
    ap.add_argument("--precision", "-p", type=int, choices=[16, 32], default=32)
    ap.add_argument("--validate-every-n-epoch", "-ve", type=int, default=1, choices=list(range(1, 20)))
    ap.add_argument("--resume", "-r", type=str)

    main(
        ap.parse_args()
    )
