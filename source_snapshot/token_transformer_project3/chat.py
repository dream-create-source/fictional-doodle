import glob
import os
import re

import torch

from common import clean_generated_answer, get_device, load_model_checkpoint, load_tokenizer_from_checkpoint


PROJECT_DIR = os.path.dirname(__file__)
CHECKPOINT_CANDIDATES = [
    os.path.join(PROJECT_DIR, "stage2_final.pt"),
    os.path.join(PROJECT_DIR, "stage1_final.pt"),
]


def list_checkpoints():
    paths = [path for path in CHECKPOINT_CANDIDATES if os.path.exists(path)]
    paths += sorted(glob.glob(os.path.join(PROJECT_DIR, "checkpoints", "*", "*.pt")))
    deduped = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def generate_reply(model, tokenizer, prompt, device, max_new_tokens, temperature, instruct_mode):
    if instruct_mode:
        prompt_text = f"User: {prompt}\nAssistant:"
    else:
        prompt_text = prompt

    prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    if not prompt_ids:
        return None

    token_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(token_ids, max_new_tokens=max_new_tokens, temperature=temperature)

    generated = tokenizer.decode(output[0].tolist()[len(prompt_ids):])
    generated = clean_generated_answer(generated)
    if not generated and not instruct_mode:
        generated = tokenizer.decode(output[0].tolist())
    return generated


def chat(model, tokenizer, instruct_mode, checkpoint_path):
    print("\n" + "═" * 60)
    print("  Project 3 Chat")
    print(f"  Model: {os.path.basename(checkpoint_path)}")
    print(f"  Mode: {'Assistant' if instruct_mode else 'Pretrain'}")
    print("═" * 60)
    print("Commands:")
    print("  /temp 0.8     -> set temperature")
    print("  /length 120   -> set output length in tokens")
    print("  /quit         -> exit")
    print("═" * 60 + "\n")

    temperature = 0.7
    length = 120

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.startswith("/quit"):
            print("Goodbye!")
            break
        if prompt.startswith("/temp"):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
            except Exception:
                print("Usage: /temp 0.8")
            continue
        if prompt.startswith("/length"):
            try:
                length = int(prompt.split()[1])
                print(f"Output length set to {length}")
            except Exception:
                print("Usage: /length 120")
            continue
        if prompt.startswith("/"):
            print("Unknown command")
            continue

        answer = generate_reply(model, tokenizer, prompt, device, length, temperature, instruct_mode)
        if answer is None:
            print("(couldn't encode prompt)\n")
            continue

        answer = re.sub(r"\s+\n", "\n", answer).strip()
        print(f"\nAI: {answer}")
        print("─" * 60 + "\n")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    checkpoints = list_checkpoints()
    if not checkpoints:
        print("No project3 checkpoints found.")
        raise SystemExit(1)

    print("\nAvailable models:")
    for index, path in enumerate(checkpoints):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{index}] {os.path.basename(path)} ({size_mb:.1f} MB)")

    if len(checkpoints) == 1:
        choice = 0
    else:
        try:
            choice = int(input(f"\nPick a model [0-{len(checkpoints)-1}]: "))
        except Exception:
            choice = 0

    checkpoint_path = checkpoints[choice]
    checkpoint, model = load_model_checkpoint(checkpoint_path, device)
    tokenizer = load_tokenizer_from_checkpoint(checkpoint)
    instruct_mode = checkpoint.get("stage") == "stage2_instruct"

    chat(model, tokenizer, instruct_mode, checkpoint_path)
