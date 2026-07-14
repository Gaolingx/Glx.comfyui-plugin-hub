import torch
import folder_paths
import comfy.utils
import comfy.lora
import logging

class INT8GroupedLora:
    """
    A simple node to stack multiple LoRAs onto a model.
    The actual application logic is intercepted by INT8ModelPatcher in int8_quant.py.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {}
        }
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_loras"
    CATEGORY = "loaders"
    DESCRIPTION = "Stacks multiple LoRAs onto an INT8 model. Actual patching is handled by INT8ModelPatcher."

    def apply_loras(self, model, **kwargs):
        model_patcher = model.clone()

        # ComfyUI's ModelPatcher.clone() builds a fresh patcher object and does
        # NOT carry over arbitrary attributes set on the source patcher (e.g.
        # the _safetensors_metadata stash from UNetLoaderINTW8A8). Without this,
        # downstream nodes like INT8ModelSave lose the source safetensors
        # metadata (int8_quantized flag, model_type, LTX2 'config', etc.) and
        # produce a corrupted/unloadable checkpoint.
        for attr in ("_safetensors_metadata", "_int8_source_metadata"):
            if hasattr(model, attr) and not hasattr(model_patcher, attr):
                try:
                    setattr(model_patcher, attr, getattr(model, attr))
                except Exception:
                    pass

        # Get key mappings from ComfyUI's framework
        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)
            
        applied_loras = []
        for i in range(1, 11):
            name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)
            
            if name and name != "None" and strength != 0:
                lora_path = folder_paths.get_full_path("loras", name)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                patch_dict = comfy.lora.load_lora(lora_data, key_map)
                model_patcher.add_patches(patch_dict, strength)
                applied_loras.append(name)
                del lora_data
        
        if applied_loras:
            logging.info(f"INT8 Grouped LoRA: Stacked {len(applied_loras)} LoRAs: {', '.join(applied_loras)}")
        
        return (model_patcher,)

NODE_CLASS_MAPPINGS = {
    "INT8GroupedLora": INT8GroupedLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8GroupedLora": "INT8 Grouped LoRA",
}