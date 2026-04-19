import glob
import os
import re

import torch
import torch.nn.functional as F

from common import clean_generated_answer, get_device, load_model_checkpoint, load_tokenizer_from_checkpoint


PROJECT_DIR = os.path.dirname(__file__)
CHECKPOINT_CANDIDATES = [
    os.path.join(PROJECT_DIR, "stage4_final.pt"),
    os.path.join(PROJECT_DIR, "stage3_final.pt"),
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


def is_domain_question(prompt):
    prompt_lower = prompt.lower()
    domain_terms = [
        "uap", "ufo", "sighting", "norea", "denebian", "probe", "quotes", "source",
        "phenomena", "object", "craft", "witness", "lights", "contact",
    ]
    question_starters = ("what", "who", "when", "where", "why", "how", "which", "describe", "explain")
    return prompt_lower.startswith(question_starters) or any(term in prompt_lower for term in domain_terms)


def generate_qa_answer(model, tokenizer, prompt, device, max_new_tokens, temperature):
    qa_prompt = f"Q: {prompt}\nA:"
    prompt_ids = tokenizer.encode(qa_prompt, add_bos=True, add_eos=False)
    if not prompt_ids:
        return None

    token_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            token_ids_crop = token_ids[:, -model.max_seq_len:]
            logits, _ = model(token_ids_crop)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

            answer = tokenizer.decode(token_ids[0].tolist()[len(prompt_ids):])
            if "\nQ:" in answer or "\n\n" in answer:
                break

    answer = tokenizer.decode(token_ids[0].tolist()[len(prompt_ids):])
    return clean_generated_answer(answer) or "(no answer generated)"


def generate_freeform(model, tokenizer, prompt, device, max_new_tokens, temperature):
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if not prompt_ids:
        return None
    token_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(token_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(output[0].tolist())


def chat(model, tokenizer, qa_mode, checkpoint_path):
    print("\n" + "═" * 60)
    print("  Project 1 Chat")
    print(f"  Model: {os.path.basename(checkpoint_path)}")
    print(f"  Mode: {'Q&A' if qa_mode else 'Freeform'}")
    print("═" * 60)
    print("Commands:")
    print("  /temp 0.8     -> set temperature")
    print("  /length 80    -> set output length in tokens")
    print("  /quit         -> exit")
    print("═" * 60 + "\n")

    temperature = 0.7
    length = 80

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
                print("Usage: /length 80")
            continue
        if prompt.startswith("/"):
            print("Unknown command")
            continue

        if qa_mode and not is_domain_question(prompt):
            print("\nAI: I'm a UAP research Q&A model. Ask me a question about the sources, sightings, objects, witnesses, or related phenomena.")
            print("─" * 60 + "\n")
            continue

        if qa_mode:
            answer = generate_qa_answer(model, tokenizer, prompt, device, length, temperature)
        else:
            answer = generate_freeform(model, tokenizer, prompt, device, length, temperature)

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
        print("No project1 checkpoints found.")
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
    qa_mode = checkpoint.get("stage") in {"stage3_qna", "stage4_rlaif"} or "qa_dataset_file" in checkpoint

    chat(model, tokenizer, qa_mode, checkpoint_path)
