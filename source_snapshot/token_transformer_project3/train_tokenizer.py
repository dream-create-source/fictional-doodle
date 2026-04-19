import os

from common import (
    TOKENIZER_CONFIG,
    build_tokenizer_from_text,
    load_corpus_from_paths,
    parse_env_paths,
    save_tokenizer,
)


PROJECT_DIR = os.path.dirname(__file__)
PROJECT2_DIR = os.path.join(PROJECT_DIR, "..", "project2", "data")

TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", os.path.join(PROJECT_DIR, "tokenizer.json"))
TOKENIZER_SOURCE_PATHS = parse_env_paths(
    os.getenv(
        "TOKENIZER_SOURCE_PATHS",
        ",".join(
            [
                os.path.join(PROJECT2_DIR, "wiki", "simplewiki-latest-pages-articles-multistream.xml"),
                os.path.join(PROJECT2_DIR, "books"),
                os.path.join(PROJECT2_DIR, "science", "arxiv_abstracts.jsonl"),
                os.path.join(PROJECT2_DIR, "instruct", "databricks-dolly-15k.jsonl"),
                os.path.join(PROJECT2_DIR, "synthetic", "ollama_synthetic.jsonl"),
            ]
        ),
    )
)
JSON_TOKENIZER_RECORDS = int(os.getenv("JSON_TOKENIZER_RECORDS", "60000"))
MAX_CHARS_PER_FILE = int(os.getenv("MAX_CHARS_PER_FILE", "12000000"))
TOTAL_CHAR_CAP = int(os.getenv("TOTAL_CHAR_CAP", "50000000"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", str(TOKENIZER_CONFIG["vocab_size"])))


text, loaded_paths = load_corpus_from_paths(
    TOKENIZER_SOURCE_PATHS,
    json_max_records=JSON_TOKENIZER_RECORDS,
    max_chars_per_file=MAX_CHARS_PER_FILE,
    total_char_cap=TOTAL_CHAR_CAP,
)
tokenizer = build_tokenizer_from_text(text, vocab_size=VOCAB_SIZE)

save_tokenizer(
    TOKENIZER_PATH,
    tokenizer,
    extra={
        "source_paths": loaded_paths,
        "char_count": len(text),
        "json_tokenizer_records": JSON_TOKENIZER_RECORDS,
        "max_chars_per_file": MAX_CHARS_PER_FILE,
        "total_char_cap": TOTAL_CHAR_CAP,
    },
)

print("Tokenizer sources:")
for path in loaded_paths:
    print(f"  - {path}")
print(f"Character sample size: {len(text):,}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
print(f"Saved tokenizer -> {TOKENIZER_PATH}")
