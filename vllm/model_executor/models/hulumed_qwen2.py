from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, Optional, List
import re

import torch
import torch.nn as nn
import numpy as np

from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import get_processor
from vllm.logger import init_logger

from .interfaces import (
    SupportsMultiModal, 
    SupportsLoRA, 
    SupportsPP, 
    MultiModalEmbeddings
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

from . import hulumed_encoder

class HulumedMlpProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_type: str = "mlp2x_gelu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match is None:
            mlp_depth = 2
        else:
            mlp_depth = int(mlp_gelu_match.group(1))
        
        modules = [nn.Linear(vision_hidden_size, text_hidden_size, bias=True)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(text_hidden_size, text_hidden_size, bias=True))
        
        self.readout = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(x)

class HulumedProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        from transformers import AutoConfig
        return self.ctx.get_hf_config(AutoConfig)
    
    def get_hf_processor(self, **kwargs):
        processor = get_processor(
            self.ctx.model_config.tokenizer,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            **kwargs,
        )
        return processor
    
    def get_image_processor(self, **kwargs):
        processor = self.get_hf_processor(**kwargs)
        if hasattr(processor, 'image_processor'):
            return processor.image_processor
        return processor
    
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

class HulumedMultiModalDataParser(MultiModalDataParser):
    def _parse_hf_inputs(
        self,
        hf_inputs: dict,
        hf_processor_mm_kwargs: Mapping[str, Any],
    ) -> MultiModalDataItems:
        pixel_values = hf_inputs.get("pixel_values")
        grid_sizes = hf_inputs.get("grid_sizes")
        merge_sizes = hf_inputs.get("merge_sizes")
        modals = hf_inputs.get("modals", ["image"])
        
        if pixel_values is None:
            return MultiModalDataItems()
        
        items = MultiModalDataItems()
        
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.ndim == 2:
                pixel_values = pixel_values.unsqueeze(0)
            elif pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
                pass
            
            if not isinstance(grid_sizes, torch.Tensor):
                grid_sizes = torch.tensor(grid_sizes) if grid_sizes is not None else None
            if grid_sizes is not None and grid_sizes.ndim == 1:
                grid_sizes = grid_sizes.unsqueeze(0)
            
            if merge_sizes is not None:
                if not isinstance(merge_sizes, torch.Tensor):
                    merge_sizes = torch.tensor(merge_sizes)
                if merge_sizes.ndim == 0:
                    merge_sizes = merge_sizes.unsqueeze(0)
            
            item_data = {
                "pixel_values": pixel_values,
                "grid_sizes": grid_sizes,
                "merge_sizes": merge_sizes
            }
            items.add_item("image", item_data)
            
        elif isinstance(pixel_values, (list, tuple)):
            for i, pv in enumerate(pixel_values):
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                
                if pv.ndim == 2:
                    pv = pv.unsqueeze(0)
                
                if isinstance(grid_sizes, (list, tuple)):
                    gs = grid_sizes[i] if i < len(grid_sizes) else None
                elif isinstance(grid_sizes, torch.Tensor):
                    gs = grid_sizes[i:i+1] if i < grid_sizes.shape[0] else None
                else:
                    gs = grid_sizes
                
                if isinstance(merge_sizes, (list, tuple)):
                    ms = merge_sizes[i] if i < len(merge_sizes) else None
                elif isinstance(merge_sizes, torch.Tensor):
                    ms = merge_sizes[i:i+1] if i < merge_sizes.shape[0] else None
                else:
                    ms = merge_sizes
                
                if gs is not None:
                    if not isinstance(gs, torch.Tensor):
                        gs = torch.tensor(gs)
                    if gs.ndim == 1:
                        gs = gs.unsqueeze(0)
                
                if ms is not None:
                    if not isinstance(ms, torch.Tensor):
                        ms = torch.tensor(ms)
                    if ms.ndim == 0:
                        ms = ms.unsqueeze(0)
                                
                item_data = {
                    "pixel_values": pv,
                    "grid_sizes": gs,
                    "merge_sizes": ms
                }
                items.add_item("image", item_data)
        
        return items

class HulumedMultiModalProcessor(BaseMultiModalProcessor):
    def _get_data_parser(self) -> MultiModalDataParser:
        return HulumedMultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: dict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        pixel_values = hf_inputs.get("pixel_values")
        grid_sizes = hf_inputs.get("grid_sizes")
        merge_sizes = hf_inputs.get("merge_sizes")
        
        if isinstance(pixel_values, torch.Tensor) and grid_sizes is not None:
            if not isinstance(grid_sizes, torch.Tensor):
                grid_sizes = torch.tensor(grid_sizes)
            
            if grid_sizes.ndim == 2:
                split_sizes = grid_sizes.prod(dim=1).tolist()
            else:
                split_sizes = [grid_sizes.prod().item()]
            if sum(split_sizes) == pixel_values.shape[0]:
                split_list = list(torch.split(pixel_values, split_sizes, dim=0))
                batched_list = [pv.unsqueeze(0) for pv in split_list]
                
                hf_inputs["pixel_values"] = batched_list
                
                if grid_sizes.ndim == 2:
                    batched_grid_sizes = [gs.unsqueeze(0) for gs in grid_sizes]
                else:
                    batched_grid_sizes = [grid_sizes.unsqueeze(0)]
                hf_inputs["grid_sizes"] = batched_grid_sizes
                
                if merge_sizes is not None:
                    if not isinstance(merge_sizes, torch.Tensor):
                        merge_sizes = torch.tensor(merge_sizes)
                    if merge_sizes.ndim == 1:
                        batched_merge_sizes = [ms.unsqueeze(0) for ms in merge_sizes]
                    else:
                        batched_merge_sizes = [merge_sizes.unsqueeze(0)]
                    hf_inputs["merge_sizes"] = batched_merge_sizes
        
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "grid_sizes": MultiModalFieldConfig.batched("image"),
            "merge_sizes": MultiModalFieldConfig.batched("image"),
        }
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        
        image_token = "<image>"
        if hasattr(hf_processor, "image_token") and not isinstance(hf_processor.image_token, property):
             image_token = hf_processor.image_token

        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        
        def get_replacement_hulumed(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item.get("grid_sizes")
            
            if grid_thw is not None and hasattr(grid_thw, 'data'):
                grid_thw = grid_thw.data
            
            merge_size = 1
            if "merge_sizes" in out_item:
                ms = out_item["merge_sizes"]
                if ms is not None:
                    if hasattr(ms, 'item'):
                        merge_size = ms.item()
                    elif isinstance(ms, torch.Tensor):
                        merge_size = ms.flatten()[0].item() if ms.numel() > 0 else 1
            
            if isinstance(grid_thw, torch.Tensor):
                grid_flat = grid_thw.flatten()
                if grid_flat.numel() >= 3:
                    t, h, w = int(grid_flat[0]), int(grid_flat[1]), int(grid_flat[2])
                elif grid_flat.numel() == 2:
                    h, w = int(grid_flat[0]), int(grid_flat[1])
                    t = 1
                else:
                    t, h, w = 1, 16, 16
                
                num_tokens = (h // merge_size) * (w // merge_size) * t
            else:
                num_tokens = 256 
            
            return [image_token_id] * num_tokens
        
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_hulumed,
            )
        ]

class HulumedDummyInputsBuilder:
    def __init__(self, info: HulumedProcessingInfo):
        self.info = info

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<image>" * num_images

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        *args, 
        **kwargs,
    ):
        num_images = mm_counts.get("image", 0)
        prompt = self.get_dummy_text(mm_counts)
        
        from PIL import Image
        images = [
            Image.new("RGB", (224, 224), color=0)
            for _ in range(num_images)
        ]
        
        class DummyInputs:
            def __init__(self, prompt, mm_data, hf_processor_mm_kwargs, tokenization_kwargs):
                self.prompt = prompt
                self.mm_data = mm_data
                self.hf_processor_mm_kwargs = hf_processor_mm_kwargs
                self.tokenization_kwargs = tokenization_kwargs
        
        return DummyInputs(
            prompt=prompt, 
            mm_data={"image": images},
            hf_processor_mm_kwargs={}, 
            tokenization_kwargs={}     
        )

@MULTIMODAL_REGISTRY.register_processor(
    HulumedMultiModalProcessor,
    info=HulumedProcessingInfo,
    dummy_inputs=HulumedDummyInputsBuilder,
)
class HulumedQwen2ForCausalLM(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP):
    
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_encoder.": "vision_encoder.",
            "model.mm_projector.": "mm_projector.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
    )
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        
        if hasattr(config, 'vision_encoder_config') and config.vision_encoder_config:
            vision_config = config.vision_encoder_config
            if not hasattr(vision_config, '_attn_implementation') or vision_config._attn_implementation is None:
                try: vision_config._attn_implementation = "flash_attention_2"
                except: pass
                try: vision_config['_attn_implementation'] = "flash_attention_2"
                except: pass

            self.vision_encoder = hulumed_encoder.HulumedVisionEncoder(
                config=vision_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision_encoder"),
            )
        else:
            self.vision_encoder = None
        
        if self.vision_encoder is not None:
            projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')
            self.mm_projector = HulumedMlpProjector(
                vision_hidden_size=config.vision_encoder_config.hidden_size,
                text_hidden_size=config.hidden_size,
                projector_type=projector_type,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
        else:
            self.mm_projector = None
        
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
    def get_language_model(self):
        return self.language_model
    
    def get_input_embeddings(self):
        if self.language_model is None:
            raise RuntimeError("language_model is None")
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        if self.language_model is None:
            raise RuntimeError("language_model is None")
        if hasattr(self.language_model, 'lm_head'):
            return self.language_model.lm_head
        elif hasattr(self.language_model, 'get_output_embeddings'):
            return self.language_model.get_output_embeddings()
        return None

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_sizes = kwargs.pop("grid_sizes", None)
        merge_sizes = kwargs.pop("merge_sizes", None)
        
        if pixel_values is None:
            return None
        
        return {
            "pixel_values": pixel_values, 
            "grid_sizes": grid_sizes,
            "merge_sizes": merge_sizes
        }
    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.vision_encoder is not None
        pixel_values = image_input["pixel_values"]
        grid_sizes = image_input["grid_sizes"]
        
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.ndim > 3:
                while pixel_values.ndim > 2 and any(s == 1 for s in pixel_values.shape[:-2]):
                    for i in range(pixel_values.ndim - 2):
                        if pixel_values.shape[i] == 1:
                            pixel_values = pixel_values.squeeze(i)
                            break
            
            if pixel_values.ndim == 3:
                batch_size = pixel_values.shape[0]
                if batch_size > 1:
                    pixel_values = [pixel_values[i] for i in range(batch_size)]
                else:
                    pixel_values = pixel_values.squeeze(0)
        
        if isinstance(grid_sizes, torch.Tensor):
            while grid_sizes.ndim > 2 and any(s == 1 for s in grid_sizes.shape[:-1]):
                for i in range(grid_sizes.ndim - 1):
                    if grid_sizes.shape[i] == 1:
                        grid_sizes = grid_sizes.squeeze(i)
                        break
            
            if grid_sizes.ndim == 3 and grid_sizes.shape[1] == 1:
                grid_sizes = grid_sizes.squeeze(1)
        
        if not isinstance(pixel_values, (list, tuple)):
            pixel_values = [pixel_values]
        
        if not isinstance(grid_sizes, (list, tuple)):
            if isinstance(grid_sizes, torch.Tensor):
                if grid_sizes.ndim == 2:
                    grid_sizes = [grid_sizes[i] for i in range(grid_sizes.shape[0])]
                else:
                    grid_sizes = [grid_sizes]
            else:
                grid_sizes = [grid_sizes]
        
        all_embeddings = []
        all_grid_sizes = []
        
        for idx, (pv, gs) in enumerate(zip(pixel_values, grid_sizes)):
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv)
            if not isinstance(gs, torch.Tensor):
                gs = torch.tensor(gs)
            
            pv = pv.to(self.vision_encoder.device)
            gs = gs.to(pv.device)
            
            while gs.ndim > 2 and any(s == 1 for s in gs.shape[:-1]):
                for i in range(gs.ndim - 1):
                    if gs.shape[i] == 1:
                        gs = gs.squeeze(i)
                        break
            
            if gs.ndim == 1:
                if gs.numel() == 3:
                    gs = gs.unsqueeze(0)
                else:
                    raise ValueError(f"Invalid grid_size shape: {gs.shape}, expected 3 elements")
            elif gs.ndim == 2:
                if gs.shape[1] != 3:
                    raise ValueError(f"Invalid grid_size shape: {gs.shape}, expected (*, 3)")
            else:
                raise ValueError(f"Invalid grid_size ndim: {gs.ndim}")
            
            while pv.ndim > 2:
                squeezed = False
                for i in range(pv.ndim - 2):
                    if pv.shape[i] == 1:
                        pv = pv.squeeze(i)
                        squeezed = True
                        break
                    if not squeezed:
                        break
            
            if pv.ndim == 3:
                if pv.shape[0] == 1:
                    pv = pv.squeeze(0)
                else:
                    raise ValueError(f"Invalid pixel_values shape: {pv.shape}")
            elif pv.ndim != 2:
                raise ValueError(f"Invalid pixel_values ndim: {pv.ndim}, expected 2")
            
            ms = torch.ones(1, dtype=torch.long, device=pv.device)
            image_embeds = self.vision_encoder(pv, gs, ms)
            image_features = self.mm_projector(image_embeds)
            
            all_embeddings.append(image_features)
            all_grid_sizes.append(gs)
        
        final_embeddings = torch.cat(all_embeddings, dim=0)
        final_grid_sizes = torch.cat(all_grid_sizes, dim=0)
        final_merge_sizes = torch.ones(final_grid_sizes.shape[0], dtype=torch.long, device=final_embeddings.device)
        
        return final_embeddings, final_grid_sizes, final_merge_sizes

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None: 
            return []
        
        image_features, grid_sizes, merge_sizes = self._process_image_input(image_input)
        
        grid_cpu = grid_sizes.cpu()
        
        if merge_sizes is None:
            merge_cpu = torch.ones(grid_cpu.shape[0], dtype=torch.long)
        else:
            merge_cpu = merge_sizes.cpu()
            if merge_cpu.ndim == 0:
                merge_cpu = merge_cpu.unsqueeze(0)
        
        num_tokens_per_image = []
        for i in range(grid_cpu.shape[0]):
            thw = grid_cpu[i]
            t, h, w = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
            
            m_val = int(merge_cpu[i].item()) if i < merge_cpu.numel() else 1
            num_tokens = (h // m_val) * (w // m_val) * t
            num_tokens_per_image.append(int(num_tokens))
        
        return list(image_features.split(num_tokens_per_image))
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        
        if intermediate_tensors is not None:
            inputs_embeds = None
        
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        return self.language_model.sample(logits, sampling_metadata)
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.vision_encoder is None:
            skip_prefixes.extend([
                "model.vision_encoder.", 
                "model.mm_projector.", 
                "vision_encoder.", 
                "mm_projector."
            ])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)