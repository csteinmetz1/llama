# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import re
import fire
import json
import torch
import itertools
import torchaudio
from llama import Llama
from typing import List

from lcap.utils import load_param_model, get_param_embeds
from lcap.graph import apply_audio_processing_graph

example_json_graph = """
{
    "nodes": [
        {
            "id": "limiter0",
            "plugin": "Limiter",
            "parameters": {
                "release_ms": "250",
                "threshold_db": "-40"
            },
            "sources" : ["input"],
            "targets" : ["compressor0"]
        },
        {
            "id": "compressor0",
            "plugin": "Compressor",
            "parameters": {
                "threshold_db": "-30", 
                "attack_ms": "50",
                "release_ms": "250",
                "ratio": "4"
            },
            "sources" : ["limiter0"],
            "targets" : ["reverb0"]
        },
        {
            "id": "compressor1",
            "plugin": "Compressor",
            "parameters": {
                "threshold_db": "-10", 
                "attack_ms": "5",
                "release_ms": "250",
                "ratio": "2"
            },
            "sources" : ["input"],
            "targets" : ["reverb0"]
        },
        {
            "id": "reverb0",
            "plugin": "Reverb",
            "parameters": {
                "room_size": "0.5",
                "damping": "0.5",
                "wet_level": "0.8",
                "dry_level": "0.5",
                "width": "1",
                "freeze_mode": "0"
            },
            "sources" : ["compressor0", "compressor1"],
            "targets" : []
        }
    ]
}
"""

prompt = f"""
The previous information provided represents the input and reference audio. 
Your task is to construct a graph that will transform the input audio into the reference audio.
To do so, you will define the graph using the JSON format. Here is an example graph: {example_json_graph}.

The effects and their parameters that are available for us are as follows. 
You can use as few or as many of these effects and combine them in a complex graph.

Compressor(threshold_db: float = 0, ratio: float = 1, attack_ms: float = 1.0, release_ms: float = 100)
Delay(delay_seconds: float = 0.5, feedback: float = 0.0, mix: float = 0.5)
Distortion(drive_db: float = 25)
Reverb(room_size: float = 0.5, damping: float = 0.5, wet_level: float = 0.33, dry_level: float = 0.4, width: float = 1.0, freeze_mode: float = 0.0)
Limiter(threshold_db: float = 0, release_ms: float = 100)
Chorus(rate_hz: float = 1.5, feedback: float = 0.25, depth: float = 3.0, centre_delay_ms: float = 40.0, mix: float = 0.5)
Phaser(rate_hz: float = 0.5, depth: float = 0.5, centre_frequency_hz: float = 1000.0, mix: float = 0.5)
LowpassFilter(cutoff_frequency_hz: float = 1000.0)
HighpassFilter(cutoff_frequency_hz: float = 50.0)
Bitcrush(bit_depth: float = 8)

The optimal JSON graph for this transformation is: """


def find_json_object(text):
    depth = 0
    start = None
    for i, char in enumerate(text):
        if char == "{":
            depth += 1
            if start is None:
                start = i
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
    return None


class Adaptor(torch.nn.Module):
    def __init__(self, audio_embed_dim: int, lm_embed_dim: int, hidden_size: int = 128):
        # create layers that will estimate the effect parameters
        self.mean = torch.nn.Sequential(
            torch.nn.Linear(audio_embed_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.log_std = torch.nn.Sequential(
            torch.nn.Linear(audio_embed_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, lm_embed_dim),
        )
        # self.log_std = torch.nn.Parameter(
        #    torch.ones(num_control_params) * 0.1, requires_grad=False
        # )

    def forward(self, input_embed: torch.Tensor, ref_embed: torch.Tensor):
        # state is the concatenation of the input and target embeddings
        state = torch.cat([input_embed, ref_embed], dim=1)

        # pass embeddings through policy
        mean = self.mean(state)
        log_std = self.log_std(state)
        std = torch.exp(log_std)

        return mean, std


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 512,
    max_batch_size: int = 4,
    n_iters: int = 100,
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
    print("Loading llama model...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print("Loading the audio model...")
    audio_model = (
        load_param_model(
            "/import/c4dm-datasets-ext/lcap/lcap/yrq0f9t1/checkpoints/epoch=217-step=476766.ckpt",
            use_gpu=True,
        ),
    )

    print("Loading audio files...")
    input_audio, sample_rate = torchaudio.load()
    ref_audio, sample_rate = torchaudio.load()

    # extract embedding from the reference audio
    input_embed = get_param_embeds(audio_model, input_audio, sample_rate)
    ref_embed = get_param_embeds(audio_model, ref_audio, sample_rate)

    # create simple models to adapt the embeddings
    adaptor = Adaptor(embed_dim=4096)

    # setup the optimization
    optimizer = torch.optim.Adam(adaptor.parameters(), lr=1e-3)

    # set the init embeddings to values from the embeddings
    input_lm_embed = generator.model.tok_embeddings(torch.tensor([0]))
    ref_lm_embed = generator.model.tok_embeddings(torch.tensor([0]))

    for n in range(n_iters):
        # adapt the embeddings with Adaptor first
        mean, std = adaptor(input_embed, ref_embed)
        action_dist = torch.distributions.Normal(mean, std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # run the forward method on the model
        results = generator.graph_generation(
            prompt,
            audio_embeds,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        result = results[0]

        # parse the JSON graph from the output string
        parsed_text = result["generation"].strip("\n")
        graph = find_json_object(parsed_text)

        # apply the audio processing graph to the input audio
        output_audio = apply_audio_processing_graph(graph, input_audio, sample_rate)

        # run the audio model on the output audio
        output_embed = get_param_embeds(audio_model, output_audio, sample_rate)

        # compute cosine similarity between the output and reference embeddings
        similarity = torch.cosine_similarity(output_embed, ref_embed, dim=-1)
        reward = similarity

        # compute the loss
        loss = (-log_prob * reward).mean()

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"reward: {reward.item():0.3f} - loss: {loss.item():0.3f}")


if __name__ == "__main__":
    fire.Fire(main)
