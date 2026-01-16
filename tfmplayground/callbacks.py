from abc import ABC, abstractmethod


class Callback(ABC):
    """ Abstract base class for callbacks."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            epoch_time (float): Time of the epoch in seconds.
            loss (float): Mean loss for the epoch.
            model: The model being trained.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Called to release any resources or perform cleanup.
        """
        pass


class BaseLoggerCallback(Callback):
    """ Abstract base class for logger callbacks. """
    pass


class ConsoleLoggerCallback(BaseLoggerCallback):
    """ Logger callback that prints epoch information to the console. """

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        print(
            f"Epoch {epoch}, "
            f"Loss = {loss:.4f}, "
            # f"Accuracy = {kwargs.get('accuracy', 'N/A')}, "
            # f"ROC AUC Error = {kwargs.get('roc_auc_error', 'N/A')}, "
            # f"ATE = {kwargs.get('ate', 'N/A')}, "
            f"Time = {epoch_time:.2f}s"
        )

    def close(self):
        """ Nothing to clean up for print logger. """
        pass


class TensorboardLoggerCallback(BaseLoggerCallback):
    """ Logger callback that logs epoch information to TensorBoard. """

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.add_scalar('Time/epoch', epoch_time, epoch)

    def close(self):
        self.writer.close()


class WandbLoggerCallback(BaseLoggerCallback):
    """ Logger callback that logs epoch information to Weights & Biases. """

    def __init__(
        self,
        project: str,
        name: str = None,
        group: str = None,
        tags: list = None,
        config: dict = None,
        log_dir: str = None,
    ):
        """
        Initializes a WandbLoggerCallback.

        Args:
            project (str): The name of the wandb project.
            name (str, optional): The name of the run. Defaults to None.
            config (dict, optional): Configuration dictionary for the run. Defaults to None.
            log_dir (str, optional): Directory to save wandb logs. Defaults to None.
        """
        try:
            import wandb
            self.wandb = wandb  # store wandb module to avoid import if not used
            wandb.init(
                project=project,
                name=name,
                group=group,
                tags=tags,
                config=config,
                dir=log_dir,
                resume="allow"
            )
            self.wandb.define_metric("epoch")
            self.wandb.define_metric("*", step_metric="epoch")

        except ImportError:
            raise ImportError("wandb is not installed. Install it with: pip install wandb") from e

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        log_dict = {
            "epoch": epoch,
            f"{kwargs['model_name']}/loss": loss,
            f"{kwargs['model_name']}/inaccuracy": 1 - kwargs.get('accuracy'),
            f"{kwargs['model_name']}/roc_auc_error": kwargs.get('roc_auc_error'),
            f"{kwargs['model_name']}/ate": kwargs.get('ate'),
            f"{kwargs['model_name']}/epoch_time": epoch_time,
        }
        self.wandb.log(log_dict)

    def close(self):
        self.wandb.finish()
