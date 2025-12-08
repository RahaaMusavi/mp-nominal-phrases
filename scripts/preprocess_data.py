# scripts/preprocess_data.py

import os
import pandas as pd
from collections import Counter, defaultdict
from conllu import parse_incr

def preprocess_conllu_folder(folder_path, output_csv="data/preprocessed/head_modifier_pairs.csv"):
    """
    Preprocess all .conllu files in a folder into a feature DataFrame for training.
    Computes token counts, PMI features, and extracts nominal head-modifier pairs.
    """
    all_tokens = []
    lemma_counts = Counter()

    # First pass: collect lemma counts
    for fname in os.listdir(folder_path):
        if not fname.endswith(".conllu"):
            continue
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                for token in tokenlist:
                    if isinstance(token['id'], int):
                        lemma = token.get('lemma', token.get('form'))
                        all_tokens.append(lemma)
                        lemma_counts[lemma] += 1

    # Second pass: extract head-modifier pairs and compute features
    records = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".conllu"):
            continue
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                id2token = {token['id']: token for token in tokenlist if isinstance(token['id'], int)}
                for token in tokenlist:
                    if isinstance(token['id'], int):
                        head_id = token['head']
                        if head_id in id2token:
                            head_token = id2token[head_id]
                            # Example features; expand as needed
                            record = {
                                'nominal_head_form': head_token.get('form', ''),
                                'modifier_form': token.get('form', ''),
                                'head_lemma': head_token.get('lemma', ''),
                                'modifier_lemma': token.get('lemma', ''),
                                'ezafe_label': token.get('feats', {}).get('Ezafe', 'no'),
                                'position': 'before' if token['id'] < head_id else 'after'
                            }
                            # Frequency-based feature example
                            record['head_freq'] = lemma_counts.get(head_token.get('lemma', ''), 0)
                            record['modifier_freq'] = lemma_counts.get(token.get('lemma', ''), 0)
                            records.append(record)

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}, {len(df)} rows.")
    return df