import logging
import re


# Gazingstars123/Anima-2.9B expand_manifest.json, revision
# 9f9cb502dbae7a616c3cc5a530633427fe735665. Each destination's source block,
# including the source copied into each inserted block.
BLOCK_SOURCES_28_TO_40 = (
    0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13,
    14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24, 24, 25, 26, 27,
)
MAIN_BLOCK = re.compile(r"^(?:diffusion_model\.blocks\.|lora_unet_blocks_)(\d+)[._]")


def remap_lora(lora, target_blocks, metadata=None):
    """Expand legacy Anima LoRAs without touching adapter or text-encoder keys.

    Only the published 28 -> 40 expansion is supported. In auto mode, require
    all 28 main blocks, not just a highest index below 28. Sparse files are
    ambiguous and remain unchanged. The safetensors metadata entry
    anima_source_blocks ("28" or "40") explicitly selects a source layout,
    including native 40-block LoRAs that happen to contain only blocks 0..27.
    """
    if target_blocks != 40:
        return lora

    matches = {key: match for key in lora if (match := MAIN_BLOCK.match(key))}
    blocks = {int(match[1]) for match in matches.values()}
    if not blocks:
        return lora

    source_blocks = (metadata or {}).get("anima_source_blocks")
    if source_blocks is None:
        if blocks != set(range(28)):
            if max(blocks) < 28:
                logging.warning("Anima LoRA source layout is ambiguous. Leaving blocks unchanged. "
                                "Set anima_source_blocks metadata to 28 or 40 to select the layout.")
            return lora
    elif source_blocks == "40":
        return lora
    elif source_blocks != "28" or max(blocks) >= 28:
        logging.warning("Anima LoRA source layout metadata is unsupported or conflicts with its keys. "
                        "Leaving blocks unchanged.")
        return lora

    destinations = [[] for _ in range(28)]
    for destination, source in enumerate(BLOCK_SOURCES_28_TO_40):
        destinations[source].append(destination)

    remapped = {}
    for key, tensor in lora.items():
        match = matches.get(key)
        if match is None:
            remapped[key] = tensor
            continue
        source = int(match[1])
        for destination in destinations[source]:
            remapped[key[:match.start(1)] + str(destination) + key[match.end(1):]] = tensor

    logging.info("Remapped Anima LoRA from 28 to 40 main blocks, including inserted blocks.")
    return remapped
