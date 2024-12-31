import argparse
import codecs
import glob
import os
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path

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
    description="Commandline interface for F5 TTS with Advanced Batch Processing.",
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

args = parser.parse_args()


def process_config(config_path, ema_model, vocoder):
    config = tomli.load(open(config_path, "rb"))

    ref_audio = config["ref_audio"]
    ref_text = config["ref_text"]
    gen_text = config["gen_text"]
    output_dir = config["output_dir"]
    output_file = f"{Path(config_path).stem}.wav"
    remove_silence = False
    speed = config["speed"]

    wave_path = Path(output_dir) / output_file

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
     # preprocess ref_audio and ref_text
    ref_audio, ref_text = preprocess_ref_audio_text(
        ref_audio, ref_text
    )

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for i, text in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")

        if not text.strip():
            continue

        text = re.sub(reg2, "", text)
        gen_text_ = text.strip()
        
        audio_segment, final_sample_rate, _ = infer_process(
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

    os.remove(ref_audio)


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
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
  
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
