from typing import Iterator, List, Literal
import pickle
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool

import datetime
from tqdm import tqdm as tqdm
from torch.utils.data import IterableDataset

class TrainIterableDatasetSingleFile(IterableDataset):
    def __init__(
        self,
        filepath_list: List[str],
        class_of_interest: List[str]
        max_process_num: int = 10,
    ) -> None:
        """the iterarable dataset used for training

        Parameters
        ----------
        filepath_list: List[str]
            the list contains the file paths of all data
        class_of_interests: List[str]
            the list contains the crop names that will be used, e.g., ["negative", "corn", "soybeans"]
            make sure the crop name is consistent with the training labels
        temporal_type: Literal["origin", "shifted"]
            either "origin" or "shifted"
            "origin" denotes to use time series from Jan 1st to Dec 31st of the target year
            "shifted" denotes to use time series from August 1st to next August 1st
        delta_date: int
            the length of the time period to make image composition, by default 8
        dataset_type: Literal["train", "validate"]
            either "train" or "validate"
        asset_names: List[str]
            the asset of the satellite to use
        min_valid_timestamp: int,
        max_process_num: int,
        """
        super().__init__()

        self.filepath_list = sorted(filepath_list)
        self.class_of_interests = class_of_interests
        self.process_num = max_process_num
        self.coi_cnt_list, self.sample_cnt = self.build_crop_cnt_list()
        if not config.customized_weight:
            self.weight_list = self.get_weight_list()
        else:
            self.weight_list = [i / sum(config.weight) for i in config.weight]
        self.metrics = config.metrics

    def get_backwards_iter_num(self):
        return 1

    @staticmethod
    def __get_sample_count(filepath):
        with open(filepath, "rb") as f:
            cur_sample_list = pickle.load(f)
        sample_cnt = len(cur_sample_list)
        crop_cnt_dict = dict()
        for sample_dict in cur_sample_list:
            label_set = set(sample_dict["crop"])
            # ["crop"] will a list of k items with the same "crop" type where k is the total number of images
            # from the sample year to July of the next year #noqa: E501

            for cur_label in label_set:
                if cur_label not in crop_cnt_dict:
                    crop_cnt_dict[cur_label] = 0
                crop_cnt_dict[cur_label] += 1

        return sample_cnt, crop_cnt_dict

    def build_crop_cnt_list(self):
        sample_cnt = 0
        cnt_list = [0 for _ in self.class_of_interests]
        result_list = []
        pool = ThreadPool(self.process_num)
        pbar = tqdm(total=len(self.filepath_list))

        def update_pbar(result):
            result_list.append(result)
            pbar.update(1)

        for cur_filepath in self.filepath_list:
            pool.apply_async(
                self.__get_sample_count,
                args=[cur_filepath],
                callback=update_pbar,
                error_callback=print,
            )
        pool.close()
        pool.join()

        for cur_sample_cnt, cur_label_cnt_dict in result_list:
            sample_cnt += cur_sample_cnt
            for cur_label, cur_label_cnt in cur_label_cnt_dict.items():
                cur_coi_idx = (
                    self.class_of_interests.index(cur_label.lower())
                    if cur_label.lower() in self.class_of_interests
                    else 0
                )
                cnt_list[cur_coi_idx] += cur_label_cnt

        return cnt_list, sample_cnt

    def __get_asset_index_list(self, cur_asset_names: List[str], exact_mode=True):
        asset_index_list = []
        if exact_mode:
            for val in self.asset_names:
                if val not in cur_asset_names:
                    return None
                asset_index_list.append(self.asset_names.index(val))
        else:
            for val in cur_asset_names:
                if val not in self.asset_names:
                    return None
                asset_index_list.append(self.asset_names.index(val))
        return asset_index_list

    def process_data(self, filepath):
        with open(filepath, "rb") as f:
            cur_data_list = pickle.load(f)

        if self.dataset_type == "train":
            data_idx_list = np.random.permutation(len(cur_data_list))
        else:
            data_idx_list = np.arange(len(cur_data_list))

        for idx in range(len(data_idx_list)):
            cur_filedict = cur_data_list[data_idx_list[idx]]
            cur_year = cur_filedict["year"]
            date_s = datetime.datetime(cur_year, 1, 1) + self.temporal_offset[0]
            date_e = datetime.datetime(cur_year, 1, 1) + self.temporal_offset[1]
            daterange_list = DateRangeGenerator.generate_daterange_with_target_range(
                [date_s, date_e], self.delta_date, True, False
            )

            cur_data_arr = cur_filedict["channel_val"]
            cur_time_arr = cur_filedict["datetime"]
            cur_scl_arr = cur_filedict["scl_val"]
            valid_mask_arr = ~np.isin(cur_scl_arr, SCL_INVALID_MASK)
            asset_names = cur_filedict["asset_names"]
            cur_data_label_arr = cur_filedict["crop"]
            asset_index_list = self.__get_asset_index_list(asset_names)
            cur_data_arr = cur_data_arr[:, asset_index_list].astype(np.float32)

            # ----- normalization and standardization ----- #
            cur_data_arr = (
                np.clip(cur_data_arr, S2_ORI_RANGE[0], S2_ORI_RANGE[1])
                / (S2_ORI_RANGE[1] - S2_ORI_RANGE[0])
                * (S2_TGT_RANGE[1] - S2_TGT_RANGE[0])
                + S2_TGT_RANGE[0]
            )
            # ----- normalization and standardization ----- #

            # ----- interpolation -----
            if config.interpolate:
                for missing_idx in np.unique(np.where(cur_data_arr == 0)[0]):
                    if missing_idx >= 2 and missing_idx <= len(cur_data_arr) - 3:
                        candi_arr = cur_data_arr[
                            missing_idx - 2 : missing_idx + 3
                        ].copy()
                        candi_arr[candi_arr == 0] = float("nan")
                        cur_data_arr[missing_idx] = np.nanmean(candi_arr, axis=0)
            # ----- interpolation ----- #

            # ----- calculate vegetation index ----- #
            data_type = "L2A"
            metric_arr = []
            for metric in self.metrics:
                metric_config = ndxi_collection["Sentinel-2"][metric]
                if "dehaze" in data_type:
                    asset_names_vi = [
                        band + "_DEHAZE" for band in metric_config["bands"]
                    ]
                else:
                    asset_names_vi = metric_config["bands"]
                cur_data_arr_vi = cur_data_arr[
                    :, self.__get_asset_index_list(asset_names_vi, exact_mode=False)
                ].astype(np.float32)
                derived_metric, _ = compute_ndxi(
                    np.moveaxis(cur_data_arr_vi, 0, 1),
                    ndxi_target=metric,
                    satellite_name="Sentinel-2",
                )
                metric_arr.append(derived_metric)
            metric_arr = np.moveaxis(np.array(metric_arr), 0, 1)
            if not config.check_stats and config.mode == "run":
                VI_STATS_MEAN_reorg = [VI_STATS_MEAN[vi] for vi in config.metrics]
                VI_STATS_STD_reorg = [VI_STATS_STD[vi] for vi in config.metrics]
                metric_arr = (metric_arr - VI_STATS_MEAN_reorg) / VI_STATS_STD_reorg
            else:
                VI_STATS_MEAN_reorg = 0
                VI_STATS_STD_reorg = 1
            # normalization seems will cause problem
            # metric_arr = (metric_arr - np.nanmin(metric_arr, axis=0)) / (
            #     np.nanmax(metric_arr, axis=0) - np.nanmin(metric_arr, axis=0)
            # )
            # ----- calculate vegetation index ----- #

            cur_data_arr = (cur_data_arr - S2_STATS_MEAN) / (S2_STATS_STD)
            dummy_arr = (
                np.zeros(len(self.asset_names), dtype=np.float32) - S2_STATS_MEAN
            ) / S2_STATS_STD
            data_x_arr_lst, mask_x_arr_lst, data_y_arr_lst, weight_y_arr_lst = (
                [],
                [],
                [],
                [],
            )
            expand_num = config.num_expand
            seeds = np.random.choice(1000, expand_num)
            for seed in seeds:
                selected_index = datetime_index_selection(
                    cur_time_arr, valid_mask_arr, daterange_list, seed=seed
                )
                tmp_data_x_arr_list = []
                tmp_data_y_arr_list = []
                tmp_valid_mask_arr_list = []
                tmp_data_y_weight_list = []
                for cur_selected_idx in selected_index:
                    cur_val = (
                        dummy_arr
                        if cur_selected_idx is None
                        else cur_data_arr[cur_selected_idx]
                    )

                    cur_val_vi = (
                        # np.nanmean(metric_arr, axis=0)
                        (np.zeros(len(config.metrics)) - VI_STATS_MEAN_reorg)
                        / VI_STATS_STD_reorg
                        if cur_selected_idx is None
                        else metric_arr[cur_selected_idx]
                    )
                    if config.feature == "optical_vi":
                        cur_val = np.concatenate([cur_val, cur_val_vi])
                    elif config.feature == "optical":
                        cur_val = cur_val
                    else:
                        cur_val = cur_val_vi

                    # ----- whether to make date with no data a "negative" label -----#
                    # cur_label_name = (
                    #     "negative"
                    #     if cur_selected_idx is None
                    #     else cur_data_label_arr[cur_selected_idx]
                    # )
                    # ----- whether to make date with no data a "negative" label -----#

                    cur_label_name = cur_data_label_arr[0]

                    cur_label_index = (
                        self.class_of_interests.index(cur_label_name.lower())
                        if cur_label_name.lower() in self.class_of_interests
                        else self.class_of_interests.index("negative")
                    )

                    # cur_valid_mask = (
                    #     False
                    #     if cur_selected_idx is None
                    #     else valid_mask_arr[cur_selected_idx]
                    # )
                    cur_valid_mask = True

                    tmp_data_x_arr_list.append(cur_val)
                    tmp_data_y_arr_list.append(cur_label_index)
                    tmp_valid_mask_arr_list.append(cur_valid_mask)
                    tmp_data_y_weight_list.append(
                        self.weight_list[cur_label_index] * cur_valid_mask
                    )

                data_x_arr = np.stack(tmp_data_x_arr_list, axis=0).astype(
                    np.float32
                )  # T * C

                data_y_arr = np.array(tmp_data_y_arr_list, dtype=np.int64)  # T
                mask_x_arr = np.array(tmp_valid_mask_arr_list, dtype=np.bool8)  # T
                # make_figure_time_series_feature(
                #     data_x_arr, filepath.split("/")[-1].split(".")[0]
                # )
                weight_y_arr = np.array(tmp_data_y_weight_list, dtype=np.float32)  # T
                weight_y_arr[: self.min_valid_datetime] = 0
                data_x_arr_lst.append(data_x_arr)
                mask_x_arr_lst.append(mask_x_arr)
                data_y_arr_lst.append(data_y_arr)
                weight_y_arr_lst.append(weight_y_arr)

            yield data_x_arr_lst, mask_x_arr_lst, data_y_arr_lst, weight_y_arr_lst

    def __len__(self):
        return self.sample_cnt

    def __iter__(self) -> Iterator:
        if self.dataset_type == "train":
            file_idx_list = np.random.permutation(len(self.filepath_list))
        else:
            cur_rank = dist.get_rank()
            world_size = dist.get_world_size()
            file_idx_list = np.arange(cur_rank, len(self.filepath_list), world_size)
        # file_per_worker = config.batch_size // worker_info.num_workers
        # for cur_filepath_idx in range(
        #     file_per_worker * work_id,
        #     min(file_per_worker * (work_id + 1), len(file_idx_list)),
        # ):
        #     cur_filepath = self.filepath_list[file_idx_list[cur_filepath_idx]]
        for cur_filepath_idx in range(len(file_idx_list)):
            cur_filepath = self.filepath_list[file_idx_list[cur_filepath_idx]]
            yield from self.process_data(cur_filepath)

    def get_class_of_interests(self):
        return self.class_of_interests

    def get_data_dict(self):
        return_dict = {
            "temporal_type": self.temporal_type,
            "delta_date": self.delta_date,
            "asset_names": self.asset_names,
        }
        return return_dict

    def get_weight_list(self):
        weight_list = [0 for _ in self.class_of_interests]
        for idx, cnt in enumerate(self.coi_cnt_list):
            if cnt != 0:
                weight_list[idx] = 1 / cnt
        sum_weight = sum(weight_list)
        for idx in range(len(weight_list)):
            weight_list[idx] = weight_list[idx] / sum_weight
        return weight_list

    def get_classification_evaluator(self, device):
        evaluator = TimeSequenceClassificationEvaluator(
            self.time_len, len(self.class_of_interests), cm_device=device
        )
        return evaluator

    def get_metrics(self):
        return self.metrics


if __name__ == "__main__":
    import os
    from torch.utils.data.dataloader import DataLoader
    import torch

    train_data_root = (
        "/NAS6/Members/chendu/temp_folder/tmp_training_pixel_dataset/hubei_test/train"
    )
    train_fpath_list = []
    for root, _, filenames in os.walk(train_data_root):
        for filename in filenames:
            if filename.endswith(".p"):
                train_fpath_list.append(os.path.join(root, filename))

    class_of_interests = ["Rice"]
    temporal_type = "origin"
    delta_date = 16

    train_dataset = TrainIterableDatasetSingleFile(
        train_fpath_list, class_of_interests, temporal_type, delta_date=delta_date
    )

    data_loader = DataLoader(train_dataset)

    for val in data_loader:
        data_x, data_x_mask, data_y, data_y_weight = val
        if torch.sum(data_y) > 1:
            print
        else:
            print
