#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube relevance analyzer: computes relevance scores based on video Title, Description,
and Captions, and writes them back to a Google Sheet.
Excludes spam topics like accommodation, festivals, and music-only tracks.
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
    if not isinstance(text, str):
        return ""
    # NFKD normalization separates combined characters into base characters and diacritics
    nkfd_form = unicodedata.normalize('NFKD', text)
    # Encode to ASCII, ignoring characters that cannot be represented, then decode back
    ascii_text = nkfd_form.encode('ASCII', 'ignore').decode('ASCII')
    return ascii_text.lower().strip()

def extract_first_sentence(text: str) -> str:
    """Extracts the first sentence from a block of text."""
    if not isinstance(text, str):
        return ""
    # Replace newlines with spaces to treat text as a single block
    cleaned_text = text.replace('\n', ' ').strip()
    # Find the first occurrence of a sentence-ending punctuation mark
    match = re.match(r'(.+?[.!?])', cleaned_text)
    return match.group(1) if match else cleaned_text

def colnum_to_letter(n: int) -> str:
    """Converts a 1-based column number to its letter representation (e.g., 1 -> A, 27 -> AA)."""
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string

# --- Spam & Content Filtering ---

SPAM_KEYWORDS = [
    'hotel', 'hôtel', 'resort', 'inn', 'accommodation', 'motel', 'lodge',
    'guest house', 'guesthouse', 'bnb',
    'apartment', 'apartments', 'studio', 'flat',
    'festival', 'music festival', 'concert', 'festivals',
    'villa', 'villas', 'holiday home', 'vacation home', 'bungalow'
]
SPAM_PATTERNS = [re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE) for keyword in SPAM_KEYWORDS]

def is_music_or_lyrics(caption_text: str) -> bool:
    """Detects if captions seem to be only for music or lyrics."""
    if not isinstance(caption_text, str):
        return False
    lines = [line.strip() for line in caption_text.split('\n') if line.strip()]
    if not lines:
        return False

    # Check for a high percentage of lines that are just tags like [Music] or [Intro]
    tag_lines = sum(1 for line in lines if re.fullmatch(r'\[.*\]', line))
    if len(lines) > 0 and (tag_lines / len(lines)) > 0.5:
        return True

    # Check for a high percentage of very short lines, typical for lyrics
    short_lines = sum(1 for line in lines if len(line.split()) <= 5)
    if len(lines) > 0 and (short_lines / len(lines)) > 0.7:
        return True

    return False

def check_spam(row: pd.Series) -> bool:
    """Checks a single DataFrame row for spam content."""
    # Combine normalized title and description for keyword search
    text_to_check = f"{row['norm_title']} {row['norm_description']}"
    if any(pattern.search(text_to_check) for pattern in SPAM_PATTERNS):
        return True
    if is_music_or_lyrics(row['Captions']):
        return True
    return False

# --- Core Logic ---

def load_config() -> dict:
    """Loads configuration from .env file."""
    env_path = Path('.') / '.env'
    if not env_path.exists():
        logger.error(f"❌ Configuration file .env not found in {env_path.resolve()}")
        sys.exit(1)
    load_dotenv(dotenv_path=env_path)
    config = {
        "sheet_id": os.getenv("SHEET_ID"),
        "worksheet_name": os.getenv("WORKSHEET_NAME"),
        "credentials_path": os.getenv("CREDENTIALS_JSON_PATH")
    }
    if not all(config.values()):
        logger.error("❌ One or more required variables (SHEET_ID, WORKSHEET_NAME, CREDENTIALS_JSON_PATH) are missing in .env")
        sys.exit(1)
    return config

def authorize_gspread(credentials_path: str) -> gspread.Client:
    """Authorizes with Google Sheets API using service account credentials."""
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
    """Loads all data from a worksheet into a pandas DataFrame."""
    logger.info(f"⬇️ Loading data from worksheet '{worksheet.title}'...")
    start_time = time.time()
    try:
        rows = worksheet.get_all_values()
        if not rows:
            logger.error("❌ Worksheet is empty.")
            sys.exit(1)
        header = [h.strip() for h in rows[0]]
        df = pd.DataFrame(rows[1:], columns=header).fillna('').astype(str)
        logger.info(f"✅ Loaded {len(df)} rows in {time.time() - start_time:.2f}s")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data from worksheet: {e}")
        sys.exit(1)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies normalization and filtering to the DataFrame."""
    logger.info("⚙️ Pre-processing data...")
    # Normalize text fields for analysis
    df['norm_title'] = df['Title'].apply(normalize_text)
    df['norm_description'] = df['Description'].apply(normalize_text)
    df['norm_captions'] = df['Captions'].apply(normalize_text)
    # Extract the main keyword (assuming it's the first in a comma-separated list)
    df['norm_keyword'] = df['Keywords'].apply(lambda k: normalize_text(k.split(',')[0]))
    # Extract the first sentence for boosted relevance scoring
    df['first_sentence'] = df['Description'].apply(extract_first_sentence)
    df['norm_first_sentence'] = df['first_sentence'].apply(normalize_text)
    # Identify spam rows
    df['is_spam'] = df.apply(check_spam, axis=1)
    logger.info(f"🔍 Found and marked {df['is_spam'].sum()} spam rows.")
    return df

def calculate_relevance_scores(df: pd.DataFrame, keyword_modifier_func=None) -> pd.Series:
    """
    Calculates relevance scores for a DataFrame.
    A keyword_modifier_func can be provided to alter keywords before comparison.
    """
    relevance_scores = pd.Series(0.0, index=df.index)
    # Filter out spam and rows without a keyword
    valid_df = df[~df['is_spam'] & (df['norm_keyword'] != '')].copy()

    # Apply keyword modification if a function is provided
    if keyword_modifier_func:
        valid_df['modified_keyword'] = valid_df.apply(keyword_modifier_func, axis=1)
        keyword_col = 'modified_keyword'
    else:
        valid_df['modified_keyword'] = valid_df['norm_keyword']
        keyword_col = 'norm_keyword'

    # Group by the (potentially modified) keyword to compute TF-IDF within each group
    grouped = valid_df.groupby(keyword_col)
    
    vectorizer = HashingVectorizer(n_features=5000, alternate_sign=False, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer()
    
    for keyword, group_df in grouped:
        indices = group_df.index
        # Create a document for each video by combining its most relevant text parts
        docs = (group_df['norm_title'] + ' ' + group_df['norm_first_sentence'] + ' ' + group_df['norm_captions']).tolist()
        
        if not any(docs):  # Skip if all documents in the group are empty
            continue
            
        # Fit and transform the documents within the group
        X_docs = tfidf_transformer.fit_transform(vectorizer.transform(docs))
        # Transform the keyword query
        X_query = tfidf_transformer.transform(vectorizer.transform([keyword]))
        
        # Calculate cosine similarity
        sim_scores = cosine_similarity(X_query, X_docs).flatten()
        
        # Apply a boost if the keyword appears in the first sentence
        boost_mask = group_df['norm_first_sentence'].str.contains(re.escape(keyword), na=False)
        sim_scores[boost_mask] = (sim_scores[boost_mask] + 0.1).clip(upper=1.0)
        
        # Assign scores back to the main series
        relevance_scores.loc[indices] = sim_scores
        
    return relevance_scores

def write_data_back(worksheet: gspread.Worksheet, df: pd.DataFrame, columns_to_write: list):
    """Writes specified DataFrame columns back to the worksheet."""
    logger.info(f"⬆️ Writing columns {columns_to_write} back to worksheet...")
    start_time = time.time()
    try:
        header = worksheet.row_values(1)
        # Check for and add new columns if they don't exist in the header
        new_cols_to_add = [col for col in columns_to_write if col not in header]
        if new_cols_to_add:
            logger.info(f"Adding new columns to sheet: {new_cols_to_add}")
            # Append columns to the end of the header
            worksheet.update(f"{colnum_to_letter(len(header) + 1)}1", [[col] for col in new_cols_to_add])
            header.extend(new_cols_to_add)

        # Update each column individually
        update_requests = []
        for col_name in columns_to_write:
            col_idx = header.index(col_name) + 1
            col_letter = colnum_to_letter(col_idx)
            range_to_update = f"{col_letter}2:{col_letter}{len(df) + 1}"
            # Prepare data in the format gspread expects: list of lists
            values = df[[col_name]].values.tolist()
            update_requests.append({'range': range_to_update, 'values': values})
        
        if update_requests:
            worksheet.batch_update(update_requests)

        logger.info(f"✅ Write complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Failed to write data back to worksheet: {e}")


def main():
    """Main execution function."""
    total_start_time = time.time()
    
    # 1. Load configuration and authorize
    config = load_config()
    gc = authorize_gspread(config['credentials_path'])
    worksheet = gc.open_by_key(config['sheet_id']).worksheet(config['worksheet_name'])

    # 2. Load and preprocess data
    df = load_data(worksheet)
    df = preprocess_data(df)

    # 3. Calculate "Relevance" score
    logger.info("🚀 Calculating standard 'Relevance' scores...")
    df['Relevance'] = calculate_relevance_scores(df)
    logger.info(f"✅ Scored { (df['Relevance'] > 0).sum() } items for 'Relevance'.")
    
    # 4. Calculate "RelevanceWithoutBeach" score with a keyword modifier
    logger.info("🚀 Calculating contextual 'RelevanceWithoutBeach' scores...")
    beach_synonyms = {'beach', 'spiaggia', 'strand', 'plage', 'playa', 'praia'}
    country_to_beach_map = {'italy': 'spiaggia', 'germany': 'strand', 'spain': 'playa', 'france': 'plage', 'portugal': 'praia'}
    
    def beach_keyword_modifier(row):
        keyword = row['norm_keyword']
        country = normalize_text(row.get('Country', ''))
        local_beach_word = country_to_beach_map.get(country, 'beach')
        
        # If any beach synonym is in the keyword, replace it with the localized one
        if any(syn in keyword for syn in beach_synonyms):
            pattern = r'\b(?:' + '|'.join(beach_synonyms) + r')\b'
            return re.sub(pattern, local_beach_word, keyword)
        # Otherwise, append the localized beach word
        return f"{keyword} {local_beach_word}"

    df['RelevanceWithoutBeach'] = calculate_relevance_scores(df, keyword_modifier_func=beach_keyword_modifier)
    logger.info(f"✅ Scored { (df['RelevanceWithoutBeach'] > 0).sum() } items for 'RelevanceWithoutBeach'.")
    
    # 5. Write results back to Google Sheet
    write_data_back(worksheet, df, ['Relevance', 'RelevanceWithoutBeach'])
    
    logger.info(f"🎉 All tasks completed in {time.time() - total_start_time:.2f}s.")


if __name__ == "__main__":
    main()
