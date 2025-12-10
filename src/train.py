import argparse

import torch
import torch.utils.tensorboard
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import UNet


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_epochs", type=int, default=50)
    return parser.parse_args()


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    image_transform = T.Compose(
        (
            T.Resize((512, 512)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        )
    )

    target_transform = T.Compose(
        (
            T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST),
            T.ToImage(),
            T.ToDtype(torch.long, scale=False),
            T.Lambda(lambda m: m.squeeze(0)),  # 1 H W -> H W
            T.Lambda(lambda m: m - 1),  # [1,3] -> [0,2]
        )
    )

    dataset = torchvision.datasets.OxfordIIITPet(
        "data/",
        split="trainval",
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=target_transform,
    )

    train_set, val_set = torch.utils.data.random_split(dataset, (0.8, 0.2))
    train_loader = DataLoader(train_set, shuffle=True, batch_size=2)
    val_loader = DataLoader(val_set, batch_size=2)
    return train_loader, val_loader


def main(num_epochs: int) -> None:
    writer = torch.utils.tensorboard.SummaryWriter()

    train_loader, val_loader = get_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    mean_loss = torchmetrics.MeanMetric().to(device)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=3).to(device)

    for epoch in tqdm(range(num_epochs), "Epoch"):
        mean_loss.reset()
        accuracy.reset()
        f1_score.reset()
        model.train()
        for x, y in tqdm(train_loader, "Training", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)
            loss: torch.Tensor = criterion(y_hat, y)
            mean_loss.update(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            y_hat_cls = y_hat.argmax(dim=1)
            accuracy.update(y_hat_cls, y)
            f1_score.update(y_hat_cls, y)

        writer.add_scalar("loss/train", mean_loss.compute(), epoch)
        writer.add_scalar("accuracy/train", accuracy.compute(), epoch)
        writer.add_scalar("f1_score/train", f1_score.compute(), epoch)

        mean_loss.reset()
        accuracy.reset()
        f1_score.reset()
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_loader, "Validating", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                mean_loss.update(loss)

                y_hat_cls = y_hat.argmax(dim=1)
                accuracy.update(y_hat_cls, y)
                f1_score.update(y_hat_cls, y)

        writer.add_scalar("loss/val", mean_loss.compute(), epoch)
        writer.add_scalar("accuracy/val", accuracy.compute(), epoch)
        writer.add_scalar("f1_score/val", f1_score.compute(), epoch)
    
    writer.close()


if __name__ == "__main__":
    args = parse_cli_args()
    main(args.num_epochs)
