from collections import defaultdict

import numpy as np
import torch
from data_utils import Span
from inference_utils import get_top_valid_spans
from metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering


class Trainer:

    def __int__(self, model: AutoModelForQuestionAnswering):
        self.model = model

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        device: torch.device,
        epoch: int,
        flag: str
    ) -> None:
        """
        One training/evaluation cycle (loop).
        Args:
            dataloader (torch.utils.data.DataLoader): dataloader.
            optimizer (torch.optim.Optimizer): optimizer.
            writer (SummaryWriter): tensorboard writer.
            device (torch.device): cpu or cuda.
            epoch (int): number of current epochs.
            flag (str): flag to process evaluate cycle or training cycle ("train"/"test")
        """

        if flag == "train":
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = []
        batch_metrics_list = defaultdict(list)

        for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"loop over {flag} batches",
        ):
            if flag == "train":
                optimizer.zero_grad()

            instances_batch = inputs.pop("instances")

            context_list, question_list = [], []
            for instance in instances_batch:
                context_list.append(instance.context)
                question_list.append(instance.question)

            inputs = inputs.to(device)
            offset_mapping_batch = inputs.pop("offset_mapping")

            if flag == "test":
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    loss = outputs.loss
            else:
                outputs = self.model(**inputs)
                loss = outputs.loss

            if flag == "train":
                loss.backward()
                optimizer.step()

            epoch_loss.append(loss.item())
            writer.add_scalar(
                f"batch loss / {flag}", loss.item(), epoch * len(dataloader) + i
            )

            with torch.no_grad():
                self.model.eval()
                outputs_inference = self.model(**inputs)
                self.model.train()

            spans_pred_batch_top_1 = get_top_valid_spans(
                context_list=context_list,
                question_list=question_list,
                prompt_mapper=dataloader.dataset.prompt_mapper,
                inputs=inputs,
                outputs=outputs_inference,
                offset_mapping_batch=offset_mapping_batch,
                n_best_size=1,
                max_answer_length=100,  # TODO: remove hardcode
            )

            # TODO: maybe move into get_top_valid_spans
            for idx in range(len(spans_pred_batch_top_1)):
                if not spans_pred_batch_top_1[idx]:
                    empty_span = Span(
                        token="",
                        label="O",  # TODO: maybe not "O" label
                        start_context_char_pos=0,
                        end_context_char_pos=0,
                    )
                    spans_pred_batch_top_1[idx] = [empty_span]

            spans_true_batch = [instance.answer for instance in instances_batch]

            batch_metrics = compute_metrics(
                spans_true_batch=spans_true_batch,
                spans_pred_batch_top_1=spans_pred_batch_top_1,
                prompt_mapper=dataloader.dataset.prompt_mapper,
            )

            for metric_name, metric_value in batch_metrics.items():
                batch_metrics_list[metric_name].append(metric_value)
                writer.add_scalar(
                    f"batch {metric_name} / {flag}",
                    metric_value,
                    epoch * len(dataloader) + i,
                )

        avg_loss = np.mean(epoch_loss)
        print(f"{flag} loss: {avg_loss}\n")
        writer.add_scalar(f"loss / {flag}", avg_loss, epoch)

        for metric_name, metric_value_list in batch_metrics_list.items():
            metric_value = np.mean(metric_value_list)
            print(f"{flag} {metric_name}: {metric_value}\n")
            writer.add_scalar(f"{metric_name} / {flag}", metric_value, epoch)
