# Smart Study Notes Generator

A mini project that accepts a paragraph and uses a pre-trained Hugging Face summarization model to generate a short summary.

## Features
- AI-generated summary
- Original word count
- Summary word count
- Percentage of text reduction
- Simple web interface using Gradio

## Model
`sshleifer/distilbart-cnn-12-6` from Hugging Face.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

## Deployment
Create a Hugging Face Space with **Gradio** as the SDK, then upload `app.py` and `requirements.txt`. The Space will build and provide a public link.
