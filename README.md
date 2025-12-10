# AI Grammar Corrector

AI-powered grammar correction tool that helps users fix grammar, spelling, and punctuation errors in English text. It takes user input, sends it to a language model, and returns a corrected, polished version of the sentence or paragraph.

## Features

- Corrects common grammar, spelling, and punctuation mistakes in English.
- Simple, clean interface for pasting or typing text.
- Shows both **original** and **corrected** text for easy comparison.
- Handles full sentences and short paragraphs instead of only single words.
- Built as a beginner-friendly NLP project that can be extended further.

## Tech Stack

- **Language / Backend:** Python  
- **Libraries / Tools:**
  - `openai` or another LLM API client for grammar correction
  - `streamlit` / `flask` / `gradio` (update based on your UI)
  - `python-dotenv` for managing API keys (optional)
- **Other:** Git, GitHub

> Update this section with your exact stack (e.g., “Flask + OpenAI API”, “Streamlit app”, etc.).

## How It Works

1. User enters a sentence or paragraph into the text box.
2. The app sends the text to an AI model with a prompt focused on grammar correction.
3. The model returns a corrected version of the text.
4. The UI displays the corrected text along with the original input for comparison.

## Getting Started

### Prerequisites

- Python 3.9+ installed
- Git installed
- An API key for your chosen language model provider (e.g., OpenAI)

### Installation

