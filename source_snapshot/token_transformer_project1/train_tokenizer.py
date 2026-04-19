import os

from common import (
    TOKENIZER_CONFIG,
    build_tokenizer_from_text,
    load_corpus_from_paths,
    parse_env_paths,
    save_tokenizer,
)


PROJECT_DIR = os.path.dirname(__file__)
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", os.path.join(PROJECT_DIR, "tokenizer.json"))
TOKENIZER_SOURCE_PATHS = parse_env_paths(
    os.getenv(
        "TOKENIZER_SOURCE_PATHS",
        f"{PROJECT_DIR}/../data/english_basic,{PROJECT_DIR}/../shakespeare.txt,{PROJECT_DIR}/../UAP sources",
    )
)
TOKENIZER_VOCAB_SIZE = int(os.getenv("TOKENIZER_VOCAB_SIZE", str(TOKENIZER_CONFIG["vocab_size"])))
JSON_TOKENIZER_RECORDS = int(os.getenv("JSON_TOKENIZER_RECORDS", "12000"))


text, loaded_paths = load_corpus_from_paths(TOKENIZER_SOURCE_PATHS, json_max_records=JSON_TOKENIZER_RECORDS)
tokenizer = build_tokenizer_from_text(text, vocab_size=TOKENIZER_VOCAB_SIZE)

save_tokenizer(
    TOKENIZER_PATH,
    tokenizer,
    extra={
        "source_paths": loaded_paths,
        "json_tokenizer_records": JSON_TOKENIZER_RECORDS,
    },
)

print(f"Tokenizer saved -> {TOKENIZER_PATH}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
print("Tokenizer sources:")
for path in loaded_paths:
    print(f"  - {path}")
