import torch
from torch import nn
from typing import Optional,Tuple,List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class PaliGemmaForConditionalGeneration(nn.module):
    def __init__(self,config: PaliGemmaConfig):
        super().__init__()
        self.config=config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model= GemmaForCasualLM(config.text_config)
        self.language_model=language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask : Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask==1), "The input cannot be padded"

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        selected_image_features= self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        image_features= self.multi_modal_projector(selected_image_features)
        input_embeds, attention_mask,position_ids= self._merge_input_ids_with_image_features(image_features,input_embeds,input_ids,attention_mask,kv_cache)

#2:52:48