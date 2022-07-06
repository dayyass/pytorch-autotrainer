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
    compute_metrics=compute_metrics,
    experiment_name=experiment_name,
)

trainer.train(
    n_epochs=n_epochs,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
)
```

## Requirements
Python >= 3.7

## Thanks
Thanks to [Artem](https://github.com/epivoca) for help with code refactoring!
