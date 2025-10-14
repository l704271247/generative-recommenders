from typing import Dict

# from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
# from generative_recommenders.modules.multitask_module import (
#     MultitaskTaskType,
#     TaskConfig,
# )
from typing import Dict
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor
import torch
from generative_recommenders.research.data.preprocessor import get_common_preprocessors

HSTU_EMBEDDING_DIM = 256

def get_embedding_conf(
    dataset_name: str = 'yy-sid',
    embedding_dim : int = HSTU_EMBEDDING_DIM) -> Dict[str, EmbeddingConfig]:
    dp = get_common_preprocessors()[dataset_name]
    feature_conf = dp.feature_conf()
    conf = {}
    for fea in feature_conf['user_fea']:
        num = dp.get_fea_unique_num(fea)
        if num>0 and dp.is_fea(fea):
            conf[fea] = EmbeddingConfig(
                num_embeddings=num,
                embedding_dim=embedding_dim,
                name=feature_conf['user_fea'][fea].get('embedding_table', f"{fea}_emb_table"),
                feature_names=[fea],
            )
    for fea in feature_conf['item_fea']:
        num = dp.get_fea_unique_num(fea)
        if num>0 and dp.is_fea(fea):
            conf[fea] = EmbeddingConfig(
                num_embeddings=num,
                embedding_dim=embedding_dim,
                name=feature_conf['item_fea'][fea].get('embedding_table', f"{fea}_emb_table"),
                feature_names=[fea],
            )
    return conf

