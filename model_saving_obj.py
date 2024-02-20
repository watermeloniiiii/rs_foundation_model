import torch
from typing import List, Optional

FMT_DATETIME = "%Y-%m-%dT%H:%M:%SZ"


class ModelSavingObject:
    example_dict = {
        "name": "model_12",
        "description": "the no.12 model for innerMongolia",
        "class_of_interests": ["negative", "soybean", "corn"],
        "model_instance": "A model instance",
        "training_stats": {
            "lr_scheduler": "A torch optim lr scheduler state",
            "optimizer": "A torch optimzer state",
            "finished_epoch_idx": 2,
        },
    }

    def __init__(
        self,
        name: str,
        model_instance,
        train_state_dict,
        description: Optional[str] = None,
    ) -> None:
        """
        Initialization of ModelSavingObject, this object will be used for model meta
        saving. While doing inference, All data should be able to extracted from
        this object. For example, the arguments to initialize a model, the model parameters.
        Which period the model is trainied. What class of interests the model can support.
        The Loss / miou / F1 score if metric_dict is not None.

        An example dict is shown as class attribute. Search example_dict for more details.

        Parameters
        ----------
        name: str
        region: str
        description: str
            a general text-based description of the model, a typical example should be "the no.3 model for henan" unless you have any specific comments
        class_of_interests : List[str]
            What class of interests the model support. Make sure the order of
            class of interests is the same as output channel order.
        model_dict : Dict
            Dictionary for model meta and parameters. Structure should same as example_dict.
        data_dict : Dict
            Dictionary for data meta. Structure should same as example_dict.
        eval_dict : Dict, optional
            Dictonary for metric, by default None
        """
        self.name = name
        self.description = description
        self.model_instance = model_instance
        self.train_state_dict = train_state_dict
        self.description = description if description else ""

    def get_train_state_dict(self):
        return self.train_state_dict

    def get_model_instance(self) -> torch.nn.Module:
        return self.model_instance

    def get_model_name(self):
        return self.name

    def get_model_description(self):
        return self.description
