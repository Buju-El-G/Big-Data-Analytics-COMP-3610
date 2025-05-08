from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import io
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load model and tokenizer once at startup
model_name = "allenai/led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/summarize', methods=['POST'])
def handle_summarization():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        summary = led_summarize(text, model, tokenizer, device)
        return jsonify({'summary': summary})
        
    except Exception as e:
        app.logger.error(f"Summarization error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def led_summarize(text, model, tokenizer, device, max_input_length=4096, max_output_length=512):
    try:
        inputs = tokenizer(
            text,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=global_attention_mask,
            max_length=max_output_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
    except Exception as e:
        raise RuntimeError(f"LED summarization failed: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
