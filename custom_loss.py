from typing import Any, Dict, Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class InfoNCELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim) -> None:
        super(InfoNCELoss, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None) -> Tensor:
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings = torch.cat(reps)  # Shape: [2 * batch_size, embedding_dim]

        # Compute similarity scores
        similarity_matrix = self.similarity_fct(embeddings, embeddings) * self.scale  # Shape: [2 * batch_size, 2 * batch_size]

        # Create labels for contrastive loss
        batch_size = len(sentence_features[0]["input_ids"])
        labels = torch.arange(2 * batch_size, device=embeddings.device)
        labels = (labels + batch_size) % (2 * batch_size)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Compute cross entropy loss
        loss = self.cross_entropy_loss(similarity_matrix, labels)

        return loss
    
    def get_config_dict(self) -> Dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
    
