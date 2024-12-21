import argparse
import codecs
import glob
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model import DiT, UNetT

# List to keep track of temporary files
temp_files = []

parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file or directory. Default=infer/examples/basic/basic.toml",
    default=os.path.join(
        files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"
    ),
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
    default=False,
)
args = parser.parse_args()

save_chunk = args.save_chunk


def process_config(config_path, ema_model, vocoder):
    config = tomli.load(open(config_path, "rb"))

    ref_audio = config["ref_audio"]
    ref_text = config["ref_text"]
    gen_text = config["gen_text"]
    # gen_file = config["gen_file"]
    gen_file = ""

    # patches for pip pkg user
    if "infer/examples/" in ref_audio:
        ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
    if "infer/examples/" in gen_file:
        gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
    if "voices" in config:
        for voice in config["voices"]:
            voice_ref_audio = config["voices"][voice]["ref_audio"]
            if "infer/examples/" in voice_ref_audio:
                config["voices"][voice]["ref_audio"] = str(
                    files("f5_tts").joinpath(f"{voice_ref_audio}")
                )

    if gen_file:
        gen_text = codecs.open(gen_file, "r", "utf-8").read()
    output_dir = config["output_dir"]
    # output_file = config["output_file"]
    output_file = f"{Path(config_path).stem}.wav"
    remove_silence = False
    speed = config["speed"]

    wave_path = Path(output_dir) / output_file

    if save_chunk:
        output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
        if not os.path.exists(output_chunk_dir):
            os.makedirs(output_chunk_dir)

    main_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        remove_silence,
        speed,
        output_dir,
        wave_path,
    )

# inference process

def main_process(
    ref_audio,
    ref_text,
    gen_text,
    ema_model,
    vocoder,
    remove_silence,
    speed,
    output_dir,
    wave_path,
):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    # if "voices" not in config:
    voices = {"main": main_voice}
    # else:
    #     voices = config["voices"]
    #     voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for i, text in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        gen_text_ = text.strip()
        ref_audio = voices[voice]["ref_audio"]
        ref_text = voices[voice]["ref_text"]
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio,
            ref_text,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type="vocos",
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
        )
        generated_audio_segments.append(audio_segment)

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)
            # Add output file to temp_files list
            temp_files.append(f.name)

    for voice in voices:
        os.remove(voices[voice]["ref_audio"])


def cleanup_temp_files():
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Removed temporary file: {temp_file}")
        except OSError as e:
            print(f"Error removing temporary file {temp_file}: {e}")


def main():
    # load vocoder
    
    vocoder_name = "vocos"
    mel_spec_type = "vocos"
    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

    vocoder = load_vocoder(
        vocoder_name=mel_spec_type,
        is_local=False,
        local_path=vocoder_local_path,
    )

    model = "F5-TTS"
    ckpt_file = ""
    vocab_file = ""

    # load models
    if model == "F5-TTS":
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        if not ckpt_file:  # path not specified, download from repo
            if vocoder_name == "vocos":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base"
                ckpt_step = 1200000
                ckpt_file = str(
                    cached_path(
                        f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"
                    )
                )
            elif vocoder_name == "bigvgan":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base_bigvgan"
                ckpt_step = 1250000
                ckpt_file = str(
                    cached_path(
                        f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"
                    )
                )

    elif model == "E2-TTS":
        assert vocoder_name == "vocos", "E2-TTS only supports vocoder vocos"
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        if ckpt_file == "":
            repo_name = "E2-TTS"
            exp_name = "E2TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(
                cached_path(
                    f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"
                )
            )

    print(f"Using {model}...")
    ema_model = load_model(
        model_cls,
        model_cfg,
        ckpt_file,
        mel_spec_type=mel_spec_type,
        vocab_file=vocab_file,
    )

    if os.path.isdir(args.config):
        toml_files = glob.glob(os.path.join(args.config, "*.toml"))
        for toml_file in toml_files:
            process_config(toml_file, ema_model, vocoder)
    else:
        process_config(args.config, ema_model, vocoder)
    # cleanup_temp_files()


if __name__ == "__main__":
    main()
