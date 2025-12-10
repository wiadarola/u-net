import torch
from torch import nn
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import torchmetrics
import torch.utils.tensorboard
from tqdm import tqdm
import argparse

from src.model import UNet


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_epochs", type=int, default=50)
    return parser.parse_args()


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    ToTensor = T.Compose((T.ToImage(), T.ToDtype(torch.float32, scale=True)))
    dataset = torchvision.datasets.OxfordIIITPet(
        "data/",
        split="trainval",
        target_types="segmentation",
        download=True,
        transforms=T.Compose((T.Resize((512, 512)), ToTensor)),
    )
    train_set, val_set = torch.utils.data.random_split(dataset, (0.8, 0.2))
    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set)
    return train_loader, val_loader


def main(num_epochs: int) -> None:
    writer = torch.utils.tensorboard.SummaryWriter()

    train_loader, val_loader = get_dataloaders()

    model = UNet(in_channels=3, out_channels=3)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    mean_loss = torchmetrics.MeanMetric()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=3)
    for epoch in tqdm(range(num_epochs), "Epoch"):
        mean_loss.reset()
        accuracy.reset()
        f1_score.reset()
        model.train()
        for x, y in tqdm(train_loader, "Training", leave=False):
            y_hat = model(x).flatten(2)
            y = y.unique(return_inverse=True)[1].flatten(1)
            loss: torch.Tensor = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mean_loss.update(loss)
            y_hat_cls = y_hat.argmax(dim=1)
            accuracy.update(y_hat_cls, y)
            f1_score.update(y_hat_cls, y)

        writer.add_scalar("train/loss", mean_loss.compute(), epoch)
        writer.add_scalar("train/accuracy", accuracy.compute(), epoch)
        writer.add_scalar("train/f1_score", f1_score.compute(), epoch)

        mean_loss.reset()
        accuracy.reset()
        f1_score.reset()
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_loader, "Validating", leave=False):
                y_hat = model(x)
                loss = criterion(y_hat, y)

                mean_loss.update(loss)
                accuracy.update(y_hat, y)
                f1_score.update(y_hat, y)

        writer.add_scalar("train/loss", mean_loss.compute(), epoch)
        writer.add_scalar("train/accuracy", accuracy.compute(), epoch)
        writer.add_scalar("train/f1_score", f1_score.compute(), epoch)


if __name__ == "__main__":
    args = parse_cli_args()
    main(args.num_epochs)
