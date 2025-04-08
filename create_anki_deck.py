import os
import openai
import csv
import re
from typing import List, Dict, Tuple
import random
import argparse
import logging
from datetime import datetime

# Set up logging
def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up file handler
    log_file = f'logs/catalan_vocab_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    logging.info("Logging system initialized")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def clean_filename(text: str) -> str:
    """Convert text to snake_case and remove special characters."""
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and special characters with underscores
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text

def process_word_batch(words: List[str]) -> Dict[str, List[Dict]]:
    """Process a batch of words at once using OpenAI."""
    logging.debug(f"Processing batch of {len(words)} words")
    words_str = '", "'.join(words)
    prompt = f"""
    For each of these Catalan words: "{words_str}"
    
    For each word, provide all its different meanings. For each meaning:
    1. A simple example sentence in Catalan using this word with this meaning
    2. The same sentence translated to Spanish
    3. The grammatical category (e.g., noun, verb, adjective)
    4. Usage tags (e.g., common, formal, informal, furniture, food, greetings, etc.)
    
    Format your response as:
    Word: [word]
    Meaning 1:
    Catalan Example: [example]
    Spanish Example: [example]
    Grammatical Category: [category]
    Usage tags: [tags]
    
    Meaning 2:
    Catalan Example: [example]
    Spanish Example: [example]
    Grammatical Category: [category]
    Usage tags: [tags]
    
    [Continue for all meanings of this word]
    
    Word: [next word]
    [Continue for all words]
    """
    
    try:
        logging.debug("Sending batch request to OpenAI")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful language tutor."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        word_sections = re.split(r'Word: ', content)[1:]  # Skip the first empty split
        
        results = {}
        for section in word_sections:
            try:
                # Get the word (first line)
                word = section.split('\n')[0].strip().strip('"')
                meanings = []
                
                # Split by "Meaning X:" to get each meaning
                meaning_sections = re.split(r'Meaning \d+:', section)[1:]
                
                for meaning_section in meaning_sections:
                    try:
                        catalan_example = re.search(r'Catalan Example: (.*?)(?:\n|$)', meaning_section).group(1).strip()
                        spanish_example = re.search(r'Spanish Example: (.*?)(?:\n|$)', meaning_section).group(1).strip()
                        grammatical_category = re.search(r'Grammatical Category: (.*?)(?:\n|$)', meaning_section).group(1).strip()
                        usage_tags = re.search(r'Usage tags: (.*?)(?:\n|$)', meaning_section).group(1).strip()
                        
                        meanings.append({
                            'catalan_example': catalan_example,
                            'spanish_example': spanish_example,
                            'grammatical_category': grammatical_category,
                            'usage_tags': usage_tags
                        })
                        logging.debug(f"Found meaning for {word}: {catalan_example}")
                    except AttributeError:
                        logging.warning(f"Could not parse meaning section for word {word}")
                        continue
                
                results[word] = meanings
                logging.info(f"Found {len(meanings)} meanings for {word}")
            except Exception as e:
                logging.error(f"Error processing section for word: {str(e)}")
                continue
        
        return results
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        return {}

def get_word_info(word: str, batch_results: Dict[str, List[Dict]] = None) -> List[Dict]:
    """Get detailed information about the word, either from batch results or individually."""
    if batch_results and word in batch_results:
        return batch_results[word]
    
    logging.debug(f"Processing word individually: {word}")
    prompt = f"""
    For the Catalan word "{word}", provide all its different meanings. For each meaning:
    1. A simple example sentence in Catalan using this word with this meaning
    2. The same sentence translated to Spanish
    3. The grammatical category (e.g., noun, verb, adjective)
    4. Usage tags (e.g., common, formal, informal, furniture, food, greetings, etc.)
    
    Format your response as:
    Meaning 1:
    Catalan Example: [example]
    Spanish Example: [example]
    Grammatical Category: [category]
    Usage tags: [tags]
    
    Meaning 2:
    Catalan Example: [example]
    Spanish Example: [example]
    Grammatical Category: [category]
    Usage tags: [tags]
    
    [Continue for all meanings]
    """
    
    try:
        logging.debug(f"Sending request to OpenAI for word: {word}")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful language tutor."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        meanings = []
        
        # Split the content by "Meaning X:" to get each meaning
        meaning_sections = re.split(r'Meaning \d+:', content)[1:]  # Skip the first empty split
        
        for section in meaning_sections:
            try:
                catalan_example = re.search(r'Catalan Example: (.*?)(?:\n|$)', section).group(1).strip()
                spanish_example = re.search(r'Spanish Example: (.*?)(?:\n|$)', section).group(1).strip()
                grammatical_category = re.search(r'Grammatical Category: (.*?)(?:\n|$)', section).group(1).strip()
                usage_tags = re.search(r'Usage tags: (.*?)(?:\n|$)', section).group(1).strip()
                
                meanings.append({
                    'catalan_example': catalan_example,
                    'spanish_example': spanish_example,
                    'grammatical_category': grammatical_category,
                    'usage_tags': usage_tags
                })
                logging.debug(f"Found meaning for {word}: {catalan_example}")
            except AttributeError:
                logging.warning(f"Could not parse meaning section for word {word}")
                continue
        
        logging.info(f"Found {len(meanings)} meanings for {word}")
        return meanings
    except Exception as e:
        logging.error(f"Error processing word {word}: {str(e)}")
        return []

def replace_word_in_text(text: str, word: str, replacement: str) -> str:
    """Replace a word in text, handling word boundaries and preserving case."""
    # Create a regex pattern that matches the word as a whole word
    pattern = r'\b' + re.escape(word) + r'\b'
    
    def replace_match(match):
        matched_word = match.group(0)
        # Preserve the case of the original word
        if matched_word[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
    
    return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

def create_cloze_text(catalan_example: str, spanish_example: str, word: str) -> str:
    """Create cloze text with c1 and c2 markers."""
    # Use word boundary-aware replacement for both sentences
    catalan_cloze = replace_word_in_text(catalan_example, word, f"{{{{c1::{word}}}}}")
    spanish_cloze = replace_word_in_text(spanish_example, word, f"{{{{c2::{word}}}}}")
    return f"{catalan_cloze}<br>{spanish_cloze}"

def create_csv_files(words: List[str], words_per_file: int = 1500, max_words: int = None):
    """Create CSV files with the specified structure."""
    if max_words:
        words = words[:max_words]
        logging.info(f"Processing {max_words} words for testing")
    else:
        logging.info(f"Processing all {len(words)} words")
    
    total_files = (len(words) + words_per_file - 1) // words_per_file
    logging.info(f"Will create {total_files} CSV files")
    
    # Process words in batches of 50
    batch_size = 50
    batch_results = {}
    
    for file_num in range(total_files):
        start_idx = file_num * words_per_file
        end_idx = min((file_num + 1) * words_per_file, len(words))
        file_words = words[start_idx:end_idx]
        
        # Format the filename with padded numbers
        start_num = start_idx + 1
        end_num = end_idx
        filename = f"catalan_vocab_{start_num:04d}-{end_num:04d}.csv"
        logging.info(f"Creating file: {filename}")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            # Write header
            writer.writerow([
                'card_id',
                'front',
                'back',
                'catalan',
                'espanol',
                'palabra',
                'categoria_gramatical',
                'usage_tags',
                'audio_file_name'
            ])
            
            total_cards = 0
            # Process words in batches
            for batch_start in range(0, len(file_words), batch_size):
                batch_end = min(batch_start + batch_size, len(file_words))
                current_batch = file_words[batch_start:batch_end]
                
                # Get batch results
                batch_results = process_word_batch(current_batch)
                
                # Process each word in the batch
                for word_idx, word in enumerate(current_batch, start=start_idx + batch_start + 1):
                    logging.info(f"Processing word {word_idx}/{len(words)}: {word}")
                    word_meanings = get_word_info(word, batch_results)
                    
                    for meaning_idx, word_info in enumerate(word_meanings, start=1):
                        audio_filename = f"{clean_filename(word_info['catalan_example'])}.mp3"
                        card_id = f"paraula_{word_idx:04d}_{meaning_idx}"
                        
                        writer.writerow([
                            card_id,  # card_id
                            create_cloze_text(word_info['catalan_example'], word_info['spanish_example'], word),  # front
                            f"[sound:{audio_filename}]",  # back
                            word_info['catalan_example'],  # catalan
                            word_info['spanish_example'],  # espanol
                            word,  # palabra
                            word_info['grammatical_category'],  # categoria_gramatical
                            word_info['usage_tags'],  # usage_tags
                            audio_filename  # audio_file_name
                        ])
                        total_cards += 1
                        logging.debug(f"Created card {card_id}")
            
            logging.info(f"Created {total_cards} cards in {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate Catalan vocabulary flashcards')
    parser.add_argument('--max-words', type=int, help='Maximum number of words to process (for testing)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    logging.info("Starting Catalan vocabulary flashcard generation...")
    
    # Read words from file
    logging.info("Reading words from file...")
    with open('final_catalan_words_list.txt', 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    logging.info(f"Read {len(words)} words from file")
    
    # Create CSV files
    create_csv_files(words, max_words=args.max_words)
    logging.info("CSV files created successfully!")

if __name__ == "__main__":
    main() 