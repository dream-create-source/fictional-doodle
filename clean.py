import re

with open('quotes.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove everything before "Raw Text" — kills the AI summary entirely
raw_start = text.find('Raw Text')
if raw_start != -1:
    text = text[raw_start + len('Raw Text'):]
    print("✓ AI Summary removed")

# 2. Remove -- Start of ... -- and -- End of ... -- lines
text = re.sub(r'--\s*(Start|End) of.*?--\n?', '', text)
print("✓ Section separators removed")

# 3. Remove HTML filename lines
text = re.sub(r'forgottenlanguages-full\..*?\.txt\n?', '', text)
print("✓ HTML filenames removed")

# 4. Remove {filename} artifacts
text = re.sub(r'\{filename\}', '', text)
print("✓ Template artifacts removed")

# 5. Replace DENIED with [REDACTED]
count = len(re.findall(r'\bDENIED\b', text))
text  = re.sub(r'\bDENIED\b', '[REDACTED]', text)
print(f"✓ Replaced {count} DENIED instances with [REDACTED]")

# 6. Remove bare URLs
count = len(re.findall(r'https?://\S+', text))
text  = re.sub(r'https?://\S+\n?', '', text)
print(f"✓ Removed {count} URLs")

# 7. Remove repetitive garbage lines
lines       = text.split('\n')
clean_lines = []
removed     = 0
for line in lines:
    words = line.split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            removed += 1
            continue
    clean_lines.append(line)
text = '\n'.join(clean_lines)
print(f"✓ Removed {removed} repetitive lines")

# 8. Clean up whitespace
text = re.sub(r'\n{3,}', '\n\n', text)
text = re.sub(r' {2,}', ' ', text)
text = text.strip()

# Save
with open('quotes_clean.txt', 'w', encoding='utf-8') as f:
    f.write(text)

# Report
original_size = len(open('quotes.txt', 'r').read())
clean_size    = len(text)
reduction     = (1 - clean_size / original_size) * 100

print(f"\n{'─'*40}")
print(f"Original:  {original_size:,} chars")
print(f"Cleaned:   {clean_size:,} chars")
print(f"Reduced:   {reduction:.1f}%")
print(f"\nFirst 300 chars of cleaned text:")
print(f"{'─'*40}")
print(text[:300])