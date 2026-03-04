from openai import OpenAI
import random
import nltk
import os
import tqdm
import json
from typing import Optional
import argparse

# Use the nltk library to get the list of easy words.
from nltk.corpus import words
nltk.download('words')

# Use OpenAI to get the clues for the words.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

def get_clue_examples(word: str) -> list:
    """
    Get 10 clue examples for the specified word from OpenRouter.
    
    Args:
        word (str): The word for which to get the clues.
    
    Returns:
        list: A list of 10 distinct clues for the word.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"I am getting clues that will be used for the game 'crosswords'. Provide 10 distinct clue sentences for the word '{word}', each in its own line without any numerical point form. Include the number of letters in the word in parentheses at the end of each clue, e.g. (5 letters)."}
            ],
            temperature=0.7,
        )
        # Extract the assistant's reply and split it into separate clues
        clues_text = response.choices[0].message.content.strip()
        clues_list = clues_text.split("\n")
        clues_list = clues_list[:10] if len(clues_list) >= 10 else clues_list
        # Limit to exactly 10 clues in case more or fewer are generated
        return dict([(f"{i+1}", clues_list[i]) for i in range(len(clues_list))])
    except Exception as e:
        return [f"An error occurred: {e}"]

def main(num_words: Optional[int] = 10):
    easy_words = random.sample(words.words("en-basic"), num_words)
    hard_words = random.sample(words.words("en"), num_words)

    easy_tag = [False] * num_words
    hard_tag = [True] * num_words

    all_words = easy_words + hard_words
    all_tags = easy_tag + hard_tag

    # Open the file in write mode
    with open("textarena/envs/single_player/crosswords/crosswords_dataset.jsonl", "w") as f:
        for word, tag in tqdm.tqdm(zip(all_words, all_tags)):
            clues = get_clue_examples(word)
            # Write each entry as a JSON object line by line
            json_line = json.dumps({"word": word, "hardcore": tag, "clues": clues})
            f.write(json_line + "\n")

    print("Dataset generated and saved to 'textarena/envs/single_player/crosswords/crosswords_dataset.jsonl'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a crossword dataset.")
    parser.add_argument("--num_words", type=int, default=10, help="Number of easy and hard words to include in the dataset.")
    args = parser.parse_args()

    # Pass the parsed num_words to the main function
    main(args.num_words)
