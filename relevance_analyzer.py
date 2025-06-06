#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube relevance checker: computes relevance scores on video Title, Description, and Captions,
excluding spam topics like accommodation, festivals, villas, and music-only tracks.
"""
import time
import re
import unicodedata
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Authorization
scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# 2. Normalization

def normalize(text: str) -> str:
    """Нормализует текст: убирает диакритики, приводит к ASCII, нижнему регистру и убирает лишние пробелы"""
    nkfd = unicodedata.normalize('NFKD', text or "")
    cleaned = nkfd.encode('ASCII', 'ignore').decode('ASCII').lower()
    return cleaned.strip()


def extract_first_sentence(text: str) -> str:
    s = (text or '').replace('\n', ' ').strip() + ' '
    m = re.match(r'(.+?[\.!?])\s', s)
    return m.group(1) if m else s.strip()


def colnum_to_letter(n: int) -> str:
    s = ''
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

# 3. Spam filter patterns
spam_keywords = [
    'hotel','hôtel','resort','inn','accommodation','motel','lodge',
    'guest house','guesthouse','bnb',
    'apartment','apartments','studio','flat',
    'festival','music festival','concert','festivals',
    'villa','villas','holiday home','vacation home','bungalow'
]
spam_patterns = [re.compile(r"\b" + re.escape(k) + r"\b") for k in spam_keywords]

# 4. Music or lyrics-only detection
def is_music_or_lyrics(capt: str) -> bool:
    lines = [l.strip() for l in capt.split('\n') if l.strip()]
    if not lines:
        return False
    tag_lines = sum(bool(re.match(r'^\[.*\]$', l)) for l in lines)
    if tag_lines / len(lines) > 0.5:
        return True
    short_lines = sum(len(l.split()) <= 5 for l in lines)
    return short_lines / len(lines) > 0.7

# 5. Read sheet
time_start = time.time()
sheet = client.open_by_key('1Byt1QLFgwYs_nybswGqYOWqMzSm6DDF_jnL08GmXMZQ').worksheet('Results')
rows = sheet.get_all_values()
hdr, data = rows[0], rows[1:]
df = pd.DataFrame(data, columns=hdr).fillna('').astype(str)
df.columns = df.columns.str.strip()
print(f"Loaded {len(df)} rows in {time.time() - time_start:.2f}s")

# 6. Prepare fields
N = len(df)
captions = df.get('Captions', pd.Series([''] * N)).tolist()
titles   = df.get('Title', pd.Series([''] * N)).tolist()
descs    = df.get('Description', pd.Series([''] * N)).tolist()
keys     = df.get('Keywords', pd.Series([''] * N)).tolist()
first_sents = [extract_first_sentence(d) for d in descs]

# Normalize text fields
norm_titles = [normalize(t) for t in titles]
norm_descs  = [normalize(d) for d in descs]
norm_first  = [normalize(fs) for fs in first_sents]
norm_caps   = [normalize(c) for c in captions]
norm_keys   = [normalize(k.split(',')[0]) for k in keys]

# 7. Filter check
def is_spam(i: int) -> bool:
    text = norm_titles[i] + ' ' + norm_descs[i]
    if any(p.search(text) for p in spam_patterns):
        return True
    if is_music_or_lyrics(captions[i]):
        return True
    return False

# 8. Relevance computation
hv = HashingVectorizer(n_features=5000, alternate_sign=False, ngram_range=(1,2))
tf = TfidfTransformer()
boost = 0.1
relevance = [0.0] * N
groups = {}
for i, k in enumerate(norm_keys):
    if not k or is_spam(i):
        continue
    if k in norm_titles[i] or k in norm_descs[i] or k in norm_caps[i]:
        groups.setdefault(k, []).append(i)
for key, idxs in groups.items():
    docs = [f"{norm_titles[i]} {norm_first[i]} {norm_caps[i]}" for i in idxs]
    X = tf.fit_transform(hv.transform(docs))
    Q = tf.transform(hv.transform([key] * len(idxs)))
    sims = cosine_similarity(Q, X).ravel()
    for i, score in zip(idxs, sims):
        if key in norm_first[i]:
            score = min(1.0, score + boost)
        relevance[i] = float(score)
print(f"Relevance done in {time.time() - time_start:.2f}s, scored {sum(1 for v in relevance if v > 0)} items")

# 9. RelevanceWithoutBeach
beach_syns = ['beach','spiaggia','strand','plaje','playa','praia']
local_map  = {'italy':'spiaggia','germany':'strand','spain':'playa','france':'plage','portugal':'praia'}
relevance_nb = [0.0] * N
groups2 = {}
for i, k in enumerate(norm_keys):
    if not k or is_spam(i):
        continue
    country = normalize(df.at[i, 'Country']) if 'Country' in df else ''
    loc = local_map.get(country, 'beach')
    if any(s in k for s in beach_syns):
        newk = re.sub(r'(?i)\b(?:' + '|'.join(beach_syns) + r')\b', loc, k)
    else:
        newk = f"{k} {loc}"
    groups2.setdefault(newk, []).append(i)
for newk, idxs in groups2.items():
    docs = [f"{norm_titles[i]} {norm_first[i]} {norm_caps[i]}" for i in idxs]
    X2 = tf.fit_transform(hv.transform(docs))
    Q2 = tf.transform(hv.transform([newk] * len(idxs)))
    sims2 = cosine_similarity(Q2, X2).ravel()
    for i, score in zip(idxs, sims2): relevance_nb[i] = float(score)
print(f"RelevanceWithoutBeach done in {time.time() - time_start:.2f}s, groups {len(groups2)}")

# 10. Write back
time_start2 = time.time()
if 'Relevance' not in df.columns:
    df['Relevance'] = relevance
else:
    df.loc[:, 'Relevance'] = relevance
if 'RelevanceWithoutBeach' not in df.columns:
    df['RelevanceWithoutBeach'] = relevance_nb
else:
    df.loc[:, 'RelevanceWithoutBeach'] = relevance_nb
hdr_vals = sheet.row_values(1)
new_cols = [col for col in ['Relevance', 'RelevanceWithoutBeach'] if col not in hdr_vals]
if new_cols:
    total_req = len(hdr_vals) + len(new_cols)
    if total_req > sheet.col_count:
        sheet.add_cols(total_req - sheet.col_count)
    for idx, col in enumerate(new_cols, start=len(hdr_vals)+1):
        sheet.update_cell(1, idx, col)
for col in ['Relevance','RelevanceWithoutBeach']:
    cidx = df.columns.get_loc(col) + 1
    rng = f"{colnum_to_letter(cidx)}2:{colnum_to_letter(cidx)}{N+1}"
    sheet.update(rng, [[v] for v in df[col]])
print(f"Write complete in {time.time() - time_start2:.2f}s")
