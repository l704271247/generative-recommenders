# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

from dataclasses import dataclass
from pyexpat import features
from typing import List

from generative_recommenders.research.data import item_features
import pandas as pd
from typing import Any

import torch

from generative_recommenders.research.data.dataset import DatasetV2 # , MultiFileDatasetV2
from generative_recommenders.research.data.item_features import ItemFeatures
from generative_recommenders.research.data.preprocessor import get_common_preprocessors


@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    ufea_num: int
    ifea_num: int
    feature_conf: Any


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
) -> RecoDataset:
    dp = get_common_preprocessors()[dataset_name]
    max_item_id = dp.expected_max_item_id()
    all_item_ids = [i for i in range(max_item_id+1)]

    if dataset_name == "yy-sid":
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
            item_fea_len = dp.max_jagged_dimension(),
            feature_conf=dp.feature_conf()
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,
            item_fea_len = dp.max_jagged_dimension(),
            feature_conf=dp.feature_conf()
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=len(all_item_ids),  # pyre-ignore [6]
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ufea_num=dp.ufea_num(),
        ifea_num=dp.ifea_num(),
        feature_conf=dp.feature_conf()
    )
