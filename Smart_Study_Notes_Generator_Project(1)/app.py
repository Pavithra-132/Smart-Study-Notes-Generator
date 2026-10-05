import gradio as gr
from transformers import pipeline

# Pre-trained Hugging Face summarization model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    if not text or not text.strip():
        return "Please enter a paragraph."

    original_words = len(text.split())

    words = text.split()
    if len(words) > 700:
        text = " ".join(words[:700])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=100,
        min_length=25,
        num_beams=4,
        early_stopping=True
    )

    result = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    summary_words = len(result.split())

    reduction = (
        (original_words - summary_words)
        / original_words
    ) * 100

    return (
        f"SUMMARY:\n{result}\n\n"
        f"Original word count: {original_words}\n"
        f"Summary word count: {summary_words}\n"
        f"Text reduction: {reduction:.2f}%"
    )
examples = [
    ["Artificial intelligence is a branch of computer science that enables machines to perform tasks that normally require human intelligence. It is used in healthcare, education, finance, transportation and many other fields. Modern AI systems can learn from data, recognize patterns, understand language and support decision making."],
    ["Cloud computing provides on-demand access to computing resources such as servers, storage, databases and software through the internet. Organizations use cloud services because they can scale resources quickly, reduce infrastructure costs and allow users to access applications from different locations."],
    ["Cybersecurity protects computers, networks, applications and data from unauthorized access and attacks. Common threats include phishing, malware, ransomware and weak passwords. Strong authentication, software updates, backups and security awareness can reduce the risk of cyber incidents."]
]

demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Enter your paragraph"),
    outputs=gr.Textbox(lines=10, label="Generated Summary & Statistics"),
    title="Smart Study Notes Generator",
    description="Enter a paragraph to generate a short AI summary and calculate word-count reduction.",
    examples=examples
)

if __name__ == "__main__":
    demo.launch()
