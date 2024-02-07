import os
from typing import List, Union
import torch

import argparse
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import analytics.crop_recognition.common_settings.model_config as config

from common.logger import logger
from analytics.crop_recognition.models.model_saving_obj import (
    ModelSavingObject,
)

from analytics.crop_recognition.evaluators.evaluator import (
    Evaluator,
)
from analytics.crop_recognition.models.base_model import (
    BaseClassificationModel,
)


def collate_fn(data):
    """
    Customized collate function that will
    """
    import numpy as np

    # iterate the batch
    for i in range(len(data)):
        (data_x, mask_x, data_y, weight_y) = data[i]
        data_x_arr = np.concatenate([data_x_arr, data_x], axis=0) if i != 0 else data_x
        mask_x_arr = np.concatenate([mask_x_arr, mask_x], axis=0) if i != 0 else mask_x
        data_y_arr = np.concatenate([data_y_arr, data_y], axis=0) if i != 0 else data_y
        weight_y_arr = (
            np.concatenate([weight_y_arr, weight_y], axis=0) if i != 0 else weight_y
        )
    data_x_arr = torch.FloatTensor(data_x_arr)
    data_y_arr = torch.LongTensor(data_y_arr)
    weight_y_arr = torch.FloatTensor(weight_y_arr)
    mask_x_arr = torch.BoolTensor(mask_x_arr)
    return data_x_arr, mask_x_arr, data_y_arr, weight_y_arr


def init_default_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()

    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])  # os.environ["SLURM_LOCALID"]
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
    else:
        rank = int(os.environ["RANK"])

    os.environ["RANK"] = str(rank)
    args.rank = rank
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu_id = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(args.gpu_id)
    return args


def ddp_setup(args):
    logger.info(
        f"Start initialization for rank {args.rank}, world_size:{args.world_size}, gpu_id:{args.gpu_id}"
    )
    if config.mode == "debug":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12316"
    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        init_method=args.dist_url,
    )
    dist.barrier()


class CropRecognitionTrainer:
    def __init__(self, save_folder: str, gpu_id: List[int], batch_size=10000) -> None:
        """
        Initializing trainer with given save_folder, gpu_id and batch_size


        Parameters
        ----------
        save_folder : str
            The folder used to save trained model parameters.
        gpu_id : List[int]
            The GPU index to be used.
        batch_size : int
            The batch size used for training.
            TODO: Batch size can be auto determined using binary search.

        Raises
        ------
        ValueError
            _description_
        """
        if not os.path.exists(save_folder):
            raise ValueError(
                f"save folder not exists:\n{save_folder}\nCreate folder before initialization"
            )

        self.save_folder = save_folder
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        # if dist.get_rank() == 0:
        #     self.summary_writter = SummaryWriter(os.path.join(self.save_folder, "runs"))

    def evaluate_test_model(
        self,
        model: DDP,
        test_loader: DataLoader,
        evaluator: Evaluator,
    ):
        """
        Evaluate the test dataset using current DDP model.

        The distributedSampler will padding to make every gpu have same size input data.
        We need to know how many data is needed for the last iteration.
        While doing evaluation, we do not all_together the output value every time.
        The communication can be slow and time consuming. Instead, we calculate
        confusion matrix in each process and sum up all confusion matrix and
        calculate the final matric results.

        Parameters
        ----------
        model : DDP
            Pytorch Distributed Data Parallel model.
        test_dataset : BaseTrainDataset
            Test Dataset.
        evaluator : ClassificationEvaluator
            Classfication evaluator. Generate MIOU, Confusion matrix, F1-Score.

        Returns
        -------
        _type_            _description_
        """

        evaluator.reset()
        model.eval()

        test_loss: torch.FloatTensor = torch.zeros(
            (1,), dtype=torch.float32, device=self.gpu_id
        )
        test_loss_weight: torch.FloatTensor = torch.zeros(
            (1,), dtype=torch.float32, device=self.gpu_id
        )

        model_class: BaseClassificationModel = model.module

        with torch.no_grad():
            for idx, (src, src_valid_mask, label, label_weight) in enumerate(
                test_loader
            ):
                src = self.put_data_to_device(src, self.gpu_id)
                src_valid_mask = self.put_data_to_device(src_valid_mask, self.gpu_id)
                label = self.put_data_to_device(label, self.gpu_id)
                label_weight = self.put_data_to_device(label_weight, self.gpu_id)

                sum_loss, sum_loss_weight = model_class.calculate_loss(
                    model, src, src_valid_mask, label, label_weight, 1
                )

                model_class.update_evaluator(
                    model, src, src_valid_mask, label, label_weight, evaluator
                )
                test_loss += sum_loss
                test_loss_weight += sum_loss_weight

        loss_tensor_list = [
            torch.zeros(test_loss.shape, dtype=test_loss.dtype).to(self.gpu_id)
            for _ in range(dist.get_world_size())
        ]
        loss_weight_tensor_list = [
            torch.zeros(test_loss_weight.shape, dtype=test_loss_weight.dtype).to(
                self.gpu_id
            )
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(loss_tensor_list, test_loss)
        dist.all_gather(loss_weight_tensor_list, test_loss_weight)

        avg_loss = sum([float(val) for val in loss_tensor_list]) / sum(
            [float(val) for val in loss_weight_tensor_list]
        )
        new_evaluator = evaluator.all_gather_ddp()
        evaluation_dict = new_evaluator.get_metric_dict()
        evaluation_dict["loss"] = avg_loss

        return evaluation_dict

    def put_data_to_device(
        self, input_data: Union[torch.FloatTensor, List[torch.FloatTensor]], device
    ):
        """
        Put data to selected device.
        input_data can be FloatTensor or list of FloatTensors

        Parameters
        ----------
        input_data : Union[torch.FloatTensor, List[torch.FloatTensor]]
            _description_
        device :
            pytorch compatible device.

        Returns
        -------
            Same data but transfer to GPU devices.
        """
        if isinstance(input_data, torch.Tensor):
            return input_data.to(device)

        elif isinstance(input_data, (list, tuple)):
            result = []
            for cur_data in input_data:
                result.append(self.put_data_to_device(cur_data, device))
            return result

    def write_scaler_rank0(self, tag, value, step):
        if dist.get_rank() == 0 and config.mode == "run":
            wandb.log({tag: value}, step=step)
        else:
            return

    def write_scalers_rank0(self, tag, scalers, step):
        if dist.get_rank() == 0:
            if isinstance(scalers, list):
                scalers = {str(idx): val for idx, val in enumerate(scalers)}
            self.summary_writter.add_scalars(tag, scalers, step)

    def train_model(
        self,
        model: DDP,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        save_model_file_name: str,
        num_epoch: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        best_loss=None,
        evaluator: Evaluator = None,
    ):
        """
        Train model with given dataset.

        Parameters
        ----------
        model : DDP
            Pytorch DDP model.
        train_dataloader : BaseTrainDataset
            Train dataset to be used.
        valid_dataloader : Optional[BaseTrainDataset]
            Valid dataset to be used, can be None.
        save_model_file_name : str
            File name of the model parameters.
        num_epoch : int
            Number of epochs of training.
        train_part : str, optional
            Either "head" or "all", by default "all".
            If head, only modules start with head will be trained.

        Raises
        ------
        ValueError
            _description_
        """
        evaluation_dict = None

        backwards_iter_num = train_loader.dataset.get_backwards_iter_num()
        scale = 1 / backwards_iter_num

        model_class: BaseClassificationModel = model.module

        for _ in range(num_epoch):
            model.train()
            optimizer.zero_grad()
            train_loss: torch.FloatTensor = torch.zeros(
                (1,), dtype=torch.float32, device=self.gpu_id
            )
            train_loss_weight: torch.FloatTensor = torch.zeros(
                (1,), dtype=torch.float32, device=self.gpu_id
            )
            for idx_iter, (src, src_valid_mask, label, label_weight) in enumerate(
                train_loader, 1
            ):
                src = self.put_data_to_device(src, self.gpu_id)
                src_valid_mask = self.put_data_to_device(src_valid_mask, self.gpu_id)
                label = self.put_data_to_device(label, self.gpu_id)
                label_weight = self.put_data_to_device(label_weight, self.gpu_id)

                sum_loss_val, sum_loss_weight = model_class.calculate_loss(
                    model, src, src_valid_mask, label, label_weight, scale
                )
                train_loss += sum_loss_val
                train_loss_weight += sum_loss_weight

                loss_tensor_list = [
                    torch.zeros(train_loss.shape, dtype=train_loss.dtype).to(
                        self.gpu_id
                    )
                    for _ in range(dist.get_world_size())
                ]
                loss_weight_tensor_list = [
                    torch.zeros(
                        train_loss_weight.shape, dtype=train_loss_weight.dtype
                    ).to(self.gpu_id)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(loss_tensor_list, train_loss)
                dist.all_gather(loss_weight_tensor_list, train_loss_weight)
                avg_loss = sum([float(val) for val in loss_tensor_list]) / sum(
                    [float(val) for val in loss_weight_tensor_list]
                )

                if idx_iter % backwards_iter_num == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    self.write_scaler_rank0(
                        "learning rate",
                        lr_scheduler.get_last_lr()[0],
                        lr_scheduler.last_epoch,
                    )
                    self.write_scaler_rank0(
                        "train iter loss",
                        avg_loss,
                        lr_scheduler.last_epoch,
                    )

                    train_loss.fill_(0)
                    train_loss_weight.fill_(0)

            if valid_loader is not None:
                evaluation_dict = self.evaluate_test_model(
                    model, valid_loader, evaluator
                )
                self.write_scaler_rank0(
                    "valid epoch loss",
                    float(evaluation_dict["loss"]),
                    lr_scheduler.last_epoch,
                )
                if "f1_score" in evaluation_dict:
                    self.write_scaler_rank0(
                        "f1 score",
                        float(evaluation_dict["f1_score"]),
                        lr_scheduler.last_epoch,
                    )
                if "crop_f1_score" in evaluation_dict:
                    for crop_idx in range(0, len(evaluation_dict["crop_f1_score"])):
                        self.write_scaler_rank0(
                            f"{config.class_of_interests[crop_idx]}_f1_score",
                            float(evaluation_dict["crop_f1_score"][crop_idx]),
                            lr_scheduler.last_epoch,
                        )

            if dist.get_rank() == 0:
                class_of_interests = train_loader.dataset.get_class_of_interests()
                model_instance = model.module
                train_data_dict = train_loader.dataset.get_data_dict()
                metrics = train_loader.dataset.get_metrics()
                train_state_dict = {
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }

                cur_saving_obj = ModelSavingObject(
                    config.name,
                    config.region,
                    config.description,
                    class_of_interests,
                    model_instance,
                    train_data_dict,
                    train_state_dict,
                    metrics,
                    evaluation_dict,
                )

                save_best_flag = False
                if valid_loader is not None:
                    cur_loss = evaluation_dict["criteria"]

                    if best_loss is None or best_loss > cur_loss:
                        best_loss = cur_loss
                        save_best_flag = True

                if save_best_flag:
                    torch.save(
                        cur_saving_obj,
                        os.path.join(
                            self.save_folder, f"{save_model_file_name}_best.pth"
                        ),
                    )
                torch.save(
                    cur_saving_obj,
                    os.path.join(
                        self.save_folder, f"{save_model_file_name}_latest.pth"
                    ),
                )


def train_main(
    args,
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    lr_scheduler,
    best_loss,
    evaluator,
    num_epoch=1000,
    save_checkpoint_name="saved_checkpoint",
):
    trainer = CropRecognitionTrainer(args.save_folder, args.gpu_id, args.batch_size)
    trainer.train_model(
        model=model,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        save_model_file_name=save_checkpoint_name,
        num_epoch=num_epoch,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        best_loss=best_loss,
        evaluator=evaluator,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    import shutil
    from torch.utils.data.dataloader import DataLoader
    from torch.optim import AdamW

    from torch.optim.lr_scheduler import OneCycleLR

    from torch.nn.parallel import DistributedDataParallel as DDP
    from common.config import DEFAULT_CACHE_DIR

    if config.mode == "run":
        args = init_default_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    if config.mode == "debug":
        args = parser.parse_args()
        args.rank = 0
        args.world_size = 1
        args.gpu_id = 0

    args.batch_size = config.batch_size
    args.save_folder = config.save_folder

    model = CustomizedClassificationTransformerDecoder(
        config.num_channel,  # can involve more channels, e.g., VIs
        len(config.class_of_interests),
        num_layers=config.num_layer,
    ).to(args.gpu_id)

    data_root = config.data_root
    local_data_root = os.path.join(DEFAULT_CACHE_DIR, config.region)

    if args.rank == 0:
        shutil.copytree(data_root, local_data_root, dirs_exist_ok=True)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

    ddp_setup(args)
    model = DDP(model)

    train_data_root = os.path.join(local_data_root, "train")
    valid_data_root = os.path.join(local_data_root, "valid")

    train_fpath_list = []
    valid_fpath_list = []
    temporal_type = "origin"

    for root, _, filenames in os.walk(train_data_root):
        for filename in filenames:
            if filename.endswith(".p"):
                train_fpath_list.append(os.path.join(root, filename))

    for root, _, filenames in os.walk(valid_data_root):
        for filename in filenames:
            if filename.endswith(".p"):
                valid_fpath_list.append(os.path.join(root, filename))

    train_dataset = TrainIterableDatasetSingleFile(
        train_fpath_list,
        config.class_of_interests,
        temporal_type,
        delta_date=config.delta_date,
        min_valid_timestamp=7,
    )
    valid_dataset = TrainIterableDatasetSingleFile(
        valid_fpath_list,
        config.class_of_interests,
        temporal_type,
        dataset_type="valid",
        delta_date=config.delta_date,
        min_valid_timestamp=7,
    )

    train_dataloader = DataLoader(
        train_dataset,
        args.batch_size,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    evaluator = train_dataset.get_classification_evaluator(device=args.gpu_id)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = OneCycleLR(
        optimizer,
        config.max_lr,
        epochs=config.num_epoch,
        steps_per_epoch=(len(train_dataloader) // 1),
        pct_start=config.pct_start,
        div_factor=config.div_factor,
    )

    if dist.get_rank() == 0 and config.mode == "run":
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=config.region,
            # Track hyperparameters and run metadata
            config={
                "region": config.region,
                "feature": config.feature,
                "class_of_interests": config.class_of_interests,
                "batch_size": config.batch_size,
                "max_lr": config.max_lr,
                "div_factor": config.div_factor,
                "pct_start": config.pct_start,
                "delta_date": config.delta_date,
                "num_epoch": config.num_epoch,
                "num_channel": config.num_channel,
                "interpolate": config.interpolate,
                "customized_weight": config.customized_weight,
                "weight": config.weight,
            },
            name=config.name,
        )

    train_main(
        args,
        model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        lr_scheduler,
        None,
        evaluator,
        config.num_epoch,
    )
    # CUDA_VISIBLE_DEVICES=6,7
    # torchrun --standalone --nnodes=1 --nproc-per-node=2 train_process.py
    # nnodes is the number of machine
    # nproc-per-node is the number of gpu to use in each machine
