from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# TODO: maybe add logging
class Trainer:
    """
    Class for training torch models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        compute_metrics: Callable,  # TODO: maybe make optional
        experiment_name: str,
    ):
        """
        Init.

        Args:
            model (torch.nn.Module): torch model.
            optimizer (torch.optim.Optimizer): torch optimizer.
            compute_metrics (Callable): compute metrics function.
        """

        self.model = model
        self.optimizer = optimizer
        self.compute_metrics = compute_metrics
        self._experiment_name = experiment_name

        self.device = model.device

        # TODO: remove hardcode
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    # TODO: maybe add valid_dataloader
    # TODO: maybe add criterion
    def train(
        self,
        n_epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
    ) -> torch.nn.Module:
        """
        Training loop.

        Args:
            n_epochs (int): number of epochs to train.
            train_dataloader (torch.utils.data.DataLoader): train dataloader.
            test_dataloader (torch.utils.data.DataLoader): test dataloader.

        Returns:
            torch.nn.Module: torch model.
        """

        for epoch in range(n_epochs):

            print(f"Epoch [{epoch+1} / {n_epochs}]\n")

            self._epoch(
                dataloader=train_dataloader,
                epoch=epoch,
                mode="train",
            )

            self._epoch(
                dataloader=test_dataloader,
                epoch=epoch,
                mode="test",
            )

        return self.model

    def _epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        mode: str,
    ) -> None:
        """
        One training / evaluation cycle (loop).

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader (train or test).
            epoch (int): number of current epoch.
            mode (str): train or test mode (available: "train", "test").
        """

        if mode == "train":
            self.model.train()
        elif mode == "test":
            self.model.eval()
        else:  # validation
            raise ValueError(f"Mode '{mode}' is not known, use 'train' or 'test'")

        epoch_loss = []
        batch_metrics_list = defaultdict(list)

        for i, (inputs, targets) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"loop over {mode} batches",
        ):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if mode == "train":
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                outputs.loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    outputs_inference = outputs  # TODO: validate

            epoch_loss.append(outputs.loss.item())
            self.writer.add_scalar(
                f"batch loss / {mode}", outputs.loss.item(), epoch * len(dataloader) + i
            )

            if mode == "train":
                with torch.no_grad():
                    self.model.eval()
                    outputs_inference = self.model(**inputs)
                    self.model.train()

            batch_metrics = self.compute_metrics(
                outputs=outputs_inference,
                targets=targets,
            )

            for metric_name, metric_value in batch_metrics.items():
                batch_metrics_list[metric_name].append(metric_value)
                self.writer.add_scalar(
                    f"batch {metric_name} / {mode}",
                    metric_value,
                    epoch * len(dataloader) + i,
                )

        avg_loss = np.mean(epoch_loss)
        print(f"{mode} loss: {avg_loss}\n")
        self.writer.add_scalar(f"loss / {mode}", avg_loss, epoch)

        for metric_name, metric_value_list in batch_metrics_list.items():
            metric_value = np.mean(metric_value_list)
            print(f"{mode} {metric_name}: {metric_value}\n")
            self.writer.add_scalar(f"{metric_name} / {mode}", metric_value, epoch)
