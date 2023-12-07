# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from llama import Llama
from typing import List

example_json_graph = """
{
    "nodes": [
        {
            "id": "gate0",
            "plugin": "Gate",
            "parameters": {
                "AttackTime": "30ms",
                "ReleaseTime": "50ms",
                "Threshold": "-40dB"
            }
        },
        {
            "id": "compressor0",
            "plugin": "Compressor",
            "parameters": {
                "AttackTime": "30ms",
                "ReleaseTime": "50ms",
                "Threshold": "-40dB",
                "Ratio": "4:1"
            }
        },
        {
            "id": "reverb0",
            "plugin": "Reverb",
            "parameters": {
                "DecayTime": "1s",
                "RoomSize": "0.5",
                "Damping": "0.5",
                "Mix": "50%"
            }
        }
    ],
    "edges": [
        {
            "source": "gate0",
            "target": "compressor0"
        },
        {
            "source": "compressor0",
            "target": "reverb0"
        }
    ]
}
"""


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 512,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompt = f"""
    The previous information provided represents the input and reference audio. 
    Your task is to construct a graph that will transform the input audio into the reference audio.
    To do so, you will define the graph usinga  JSON format. Here is an example graph: {example_json_graph}.
    
    The effects and their parameters that are available for us are as follows. 
    You can use as few or as many of these effects and combine them in a complex graph.

    Compressor(threshold_db: float = 0, ratio: float = 1, attack_ms: float = 1.0, release_ms: float = 100)
    Delay(delay_seconds: float = 0.5, feedback: float = 0.0, mix: float = 0.5)
    Distortion(drive_db: float = 25)
    Reverb(room_size: float = 0.5, damping: float = 0.5, wet_level: float = 0.33, dry_level: float = 0.4, width: float = 1.0, freeze_mode: float = 0.0)
    
    The optimal JSON graph for this transformation is: """

    input_embed = torch.rand(1, 4096)
    ref_embed = torch.rand(1, 4096)

    results = generator.graph_generation(
        prompt,
        input_embed,
        ref_embed,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for result in results:
        parsed = result["generation"].strip("\n")
        print(f"> {parsed}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
