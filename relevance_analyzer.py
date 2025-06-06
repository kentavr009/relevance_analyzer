#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic YouTube Relevance Analyzer: computes relevance scores for videos based on
their textual content (Title, Description, Captions) against a given keyword.
The script reads data from a Google Sheet, applies configurable spam filters and
contextual keyword modifications, and writes the scores back to the sheet.
"""
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import gspread
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def normalize_text(text: str) -> str:
    """Normalizes text: removes diacritics, converts to ASCII, lowercase, and strips whitespace."""
    if not isinstance(text, str): return ""
    nkfd_form = unicodedata.normalize('NFKD', text)
    ascii_text = nkfd_form.encode('ASCII', 'ignore').decode('ASCII')
    return ascii_text.lower().strip()

def extract_first_sentence(text: str) -> str:
    """Extracts the first sentence from a block of text."""
    if not isinstance(text, str): return ""
    cleaned_text = text.replace('\n', ' ').strip()
    match = re.match(r'(.+?[.!?])', cleaned_text)
    return match.group(1) if match else cleaned_text

def colnum_to_letter(n: int) -> str:
    """Converts a 1-based column number to its letter representation (e.g., 1 -> A, 27 -> AA)."""
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string

# --- Content Filtering Logic ---

def is_music_or_lyrics(caption_text: str) -> bool:
    """Detects if captions seem to be only for music or lyrics."""
    if not isinstance(caption_text, str): return False
    lines = [line.strip() for line in caption_text.split('\n') if line.strip()]
    if not lines: return False
    tag_lines = sum(1 for line in lines if re.fullmatch(r'\[.*\]', line))
    if len(lines) > 0 and (tag_lines / len(lines)) > 0.5: return True
    short_lines = sum(1 for line in lines if len(line.split()) <= 5)
    if len(lines) > 0 and (short_lines / len(lines)) > 0.7: return True
    return False

def check_spam(row: pd.Series, spam_patterns: list) -> bool:
    """Checks a single DataFrame row for spam content using configurable patterns."""
    text_to_check = f"{row['norm_title']} {row['norm_description']}"
    if any(pattern.search(text_to_check) for pattern in spam_patterns):
        return True
    if is_music_or_lyrics(row.get('Captions', '')):
        return True
    return False

# --- Core Logic ---

def load_config() -> dict:
    """Loads and validates configuration from .env file."""
    env_path = Path('.') / '.env'
    if not env_path.exists():
        logger.error(f"❌ Configuration file .env not found in {env_path.resolve()}")
        sys.exit(1)
    load_dotenv(dotenv_path=env_path)
    
    config = {
        "sheet_id": os.getenv("SHEET_ID"),
        "worksheet_name": os.getenv("WORKSHEET_NAME"),
        "credentials_path": os.getenv("CREDENTIALS_JSON_PATH"),
        "context_column": os.getenv("CONTEXT_COLUMN_NAME", "Country"),
        "spam_keywords": [k.strip() for k in os.getenv("SPAM_KEYWORDS", "").split(',') if k.strip()],
        "context_trigger_words": [k.strip() for k in os.getenv("CONTEXT_TRIGGER_WORDS", "").split(',') if k.strip()],
        "context_default_append": os.getenv("CONTEXT_DEFAULT_APPEND_WORD", ""),
        "context_replacement_map": {}
    }

    # Parse the replacement map from .env (e.g., CONTEXT_REPLACEMENT_italy=word1)
    for key, value in os.environ.items():
        if key.startswith("CONTEXT_REPLACEMENT_"):
            context_key = key.replace("CONTEXT_REPLACEMENT_", "").lower()
            config["context_replacement_map"][context_key] = value.strip()
            
    if not all([config["sheet_id"], config["worksheet_name"], config["credentials_path"]]):
        logger.error("❌ One or more required variables (SHEET_ID, WORKSHEET_NAME, CREDENTIALS_JSON_PATH) are missing in .env")
        sys.exit(1)
    
    return config

def authorize_gspread(credentials_path: str) -> gspread.Client:
    """Authorizes with Google Sheets API."""
    logger.info("🔐 Authorizing with Google Sheets API...")
    try:
        return gspread.service_account(filename=credentials_path)
    except FileNotFoundError:
        logger.error(f"❌ Credentials file not found at: {credentials_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to authorize: {e}")
        sys.exit(1)

def load_data(worksheet: gspread.Worksheet) -> pd.DataFrame:
    """Loads data from a worksheet into a DataFrame."""
    logger.info(f"⬇️ Loading data from worksheet '{worksheet.title}'...")
    start_time = time.time()
    try:
        rows = worksheet.get_all_values()
        if not rows: logger.error("❌ Worksheet is empty."); sys.exit(1)
        header = [h.strip() for h in rows[0]]
        df = pd.DataFrame(rows[1:], columns=header).fillna('').astype(str)
        logger.info(f"✅ Loaded {len(df)} rows in {time.time() - start_time:.2f}s")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data from worksheet: {e}"); sys.exit(1)

def preprocess_data(df: pd.DataFrame, spam_keywords: list) -> pd.DataFrame:
    """Applies normalization and filtering."""
    logger.info("⚙️ Pre-processing data...")
    df['norm_title'] = df['Title'].apply(normalize_text)
    df['norm_description'] = df['Description'].apply(normalize_text)
    df['norm_captions'] = df.get('Captions', pd.Series([''] * len(df))).apply(normalize_text)
    df['norm_keyword'] = df['Keywords'].apply(lambda k: normalize_text(k.split(',')[0]))
    df['first_sentence'] = df['Description'].apply(extract_first_sentence)
    df['norm_first_sentence'] = df['first_sentence'].apply(normalize_text)
    
    spam_patterns = [re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE) for keyword in spam_keywords]
    df['is_spam'] = df.apply(check_spam, axis=1, spam_patterns=spam_patterns)
    logger.info(f"🔍 Found and marked {df['is_spam'].sum()} spam rows.")
    return df

def calculate_relevance_scores(df: pd.DataFrame, keyword_modifier_func=None) -> pd.Series:
    """Calculates relevance scores, with an optional keyword modification function."""
    relevance_scores = pd.Series(0.0, index=df.index)
    valid_df = df[~df['is_spam'] & (df['norm_keyword'] != '')].copy()

    keyword_col = 'modified_keyword'
    if keyword_modifier_func:
        valid_df[keyword_col] = valid_df.apply(keyword_modifier_func, axis=1)
    else:
        valid_df[keyword_col] = valid_df['norm_keyword']

    grouped = valid_df.groupby(keyword_col)
    vectorizer = HashingVectorizer(n_features=5000, alternate_sign=False, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer()
    
    for keyword, group_df in grouped:
        indices = group_df.index
        docs = (group_df['norm_title'] + ' ' + group_df['norm_first_sentence'] + ' ' + group_df['norm_captions']).tolist()
        if not any(docs): continue
            
        X_docs = tfidf_transformer.fit_transform(vectorizer.transform(docs))
        X_query = tfidf_transformer.transform(vectorizer.transform([keyword]))
        sim_scores = cosine_similarity(X_query, X_docs).flatten()
        boost_mask = group_df['norm_first_sentence'].str.contains(re.escape(keyword), na=False)
        sim_scores[boost_mask] = (sim_scores[boost_mask] + 0.1).clip(upper=1.0)
        relevance_scores.loc[indices] = sim_scores
        
    return relevance_scores

def create_contextual_modifier(config: dict):
    """Creates and returns a function that modifies keywords based on context rules."""
    trigger_words = set(config['context_trigger_words'])
    replacement_map = config['context_replacement_map']
    default_append = config['context_default_append']
    context_col = config['context_column']

    if not any([trigger_words, default_append]):
        return None # No contextual logic needed

    def modifier(row):
        keyword = row['norm_keyword']
        context_value = normalize_text(row.get(context_col, ''))
        replacement_word = replacement_map.get(context_value, default_append)

        if any(trigger in keyword for trigger in trigger_words):
            pattern = r'\b(?:' + '|'.join(re.escape(t) for t in trigger_words) + r')\b'
            return re.sub(pattern, replacement_word, keyword, flags=re.IGNORECASE).strip()
        
        return f"{keyword} {replacement_word}".strip()

    return modifier

def write_data_back(worksheet: gspread.Worksheet, df: pd.DataFrame, columns_to_write: list):
    """Writes specified columns back to the worksheet."""
    logger.info(f"⬆️ Writing columns {columns_to_write} back to worksheet...")
    start_time = time.time()
    try:
        header = worksheet.row_values(1)
        new_cols_to_add = [col for col in columns_to_write if col not in header]
        if new_cols_to_add:
            logger.info(f"Adding new columns to sheet: {new_cols_to_add}")
            worksheet.append_cols([ [col] for col in new_cols_to_add ])
            header.extend(new_cols_to_add)

        update_requests = []
        for col_name in columns_to_write:
            col_idx = header.index(col_name) + 1
            col_letter = colnum_to_letter(col_idx)
            range_to_update = f"{col_letter}2:{col_letter}{len(df) + 1}"
            values = df[[col_name]].fillna(0.0).values.tolist()
            update_requests.append({'range': range_to_update, 'values': values})
        
        if update_requests:
            worksheet.batch_update(update_requests)
        logger.info(f"✅ Write complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Failed to write data back to worksheet: {e}")

def main():
    """Main execution function."""
    total_start_time = time.time()
    
    config = load_config()
    gc = authorize_gspread(config['credentials_path'])
    worksheet = gc.open_by_key(config['sheet_id']).worksheet(config['worksheet_name'])

    df = load_data(worksheet)
    df = preprocess_data(df, config['spam_keywords'])

    logger.info("🚀 Calculating standard 'Relevance' scores...")
    df['Relevance'] = calculate_relevance_scores(df)
    logger.info(f"✅ Scored {(df['Relevance'] > 0).sum()} items for 'Relevance'.")
    
    contextual_modifier = create_contextual_modifier(config)
    if contextual_modifier:
        logger.info("🚀 Calculating 'RelevanceContextual' scores with keyword modification...")
        df['RelevanceContextual'] = calculate_relevance_scores(df, keyword_modifier_func=contextual_modifier)
        logger.info(f"✅ Scored {(df['RelevanceContextual'] > 0).sum()} items for 'RelevanceContextual'.")
        columns_to_write = ['Relevance', 'RelevanceContextual']
    else:
        logger.info("ℹ️ Contextual scoring rules not configured. Skipping.")
        columns_to_write = ['Relevance']
    
    write_data_back(worksheet, df, columns_to_write)
    
    logger.info(f"🎉 All tasks completed in {time.time() - total_start_time:.2f}s.")

if __name__ == "__main__":
    main()
