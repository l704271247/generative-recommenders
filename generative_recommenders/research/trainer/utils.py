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

HSTU_EMBEDDING_DIM = 256
HASH_SIZE = 20000

def get_embedding_conf(
    hash_size : int = HASH_SIZE,
    embedding_dim : int = HSTU_EMBEDDING_DIM) -> Dict[str, EmbeddingConfig]:
    conf = (
        {
            "movie_id": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="movie_id_emb_table",
                feature_names=["movie_id"],
            ),
            "sex": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="sex_emb_table",
                feature_names=["sex"],
            ),
            "age_group": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="age_group_emb_table",
                feature_names=["age_group"],
            ),
            "occupation": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="occupation_emb_table",
                feature_names=["occupation"],
            ),
            "zip_code": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="zip_code_emb_table",
                feature_names=["zip_code"],
            ),
            "genres": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="genres_emb_table",
                feature_names=["genres"],
            ),
            "title": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="title_emb_table",
                feature_names=["title"],
            ),
            "year": EmbeddingConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="year_emb_table",
                feature_names=["year"],
            ),
        }
    )
    return conf

