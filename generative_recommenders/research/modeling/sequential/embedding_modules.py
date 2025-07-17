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

import abc
from typing import Dict, List, Optional, Tuple, Union
import torch

from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor,JaggedTensor

from generative_recommenders.research.modeling.initialization import truncated_normal

class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class MyEmbeddingCollection(EmbeddingCollection):

    def forward(
        self,
        jt_dict: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        """
        Run the EmbeddingBagCollection forward pass. This method takes in a `KeyedJaggedTensor`
        and returns a `Dict[str, JaggedTensor]`, which is the result of the individual embeddings for each feature.

        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """

        feature_embeddings: Dict[str, JaggedTensor] = {}
        for i, emb_module in enumerate(self.embeddings.values()):
            feature_names = self._feature_names[i]
            embedding_names = self._embedding_names_by_table[i]
            for j, embedding_name in enumerate(embedding_names):
                feature_name = feature_names[j]
                if feature_name not in jt_dict:
                    continue
                f = jt_dict[feature_name]
                lookup = emb_module(
                    input=f.values(),
                ).float()
                feature_embeddings[embedding_name] = JaggedTensor(
                    values=lookup,
                    lengths=f.lengths(),
                    weights=f.values() if self._need_indices else None,
                )
        return feature_embeddings


class MultiEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        conf: EmbeddingConfig,
        embedding_dim: int,
        device="cpu"
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._conf = conf
        self._tables = MyEmbeddingCollection(
            tables=list(conf.values()),
            need_indices=False,
            device=device,
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"MultiEmbeddingModule_d{self._embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_emb_table" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_embeddings(self, ids: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        jt_dict = {}
        for key, val in ids.items():
            jt_dict[key] = JaggedTensor.from_dense([val])
        raw_res = self._tables(jt_dict)
        res = {}
        for key, val in raw_res.items():
            res[key] = JaggedTensor.to_dense(val)[0]
        return res

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
