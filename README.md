# pytorch-autotrainer
Wrapper for PyTorch model training.

## Installation
```
pip install pytorch-autotrainer
```

## Usage
```python3
from pytorch_autotrainer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    compute_metrics=compute_metrics,  # read below
    experiment_name=experiment_name,
)

trainer.train(
    n_epochs=n_epochs,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
)
```

### Compute Metrics
Trainer has `compute_metrics` parameter from metrics calculation.
`compute_metrics` argument should be a callable (function or class) with the following interface:

```python3
compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, int]
```

It takes model outputs and targets and returns dictionary from metric name to metric value.

## Requirements
Python >= 3.7

## Thanks
Thanks to [Artem](https://github.com/epivoca) for help with code refactoring!
