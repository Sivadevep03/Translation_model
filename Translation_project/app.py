from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

app = Flask(__name__)

# Load the translation model and tokenizer
model_name = "Helsinki-NLP/opus-mt-de-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        german_text = data['text']
        
        # Tokenize and translate
        inputs = tokenizer(german_text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs, return_dict_in_generate=True, output_attentions=True)
        
        # Get the translated text
        translated_text = tokenizer.batch_decode(translated.sequences, skip_special_tokens=True)[0]
        
        # Process attention weights for visualization
        attention_weights = translated.attentions[0].mean(dim=1).mean(dim=0).detach().numpy()
        
        # Get tokens for visualization
        source_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        target_tokens = tokenizer.convert_ids_to_tokens(translated.sequences[0])
        
        return jsonify({
            'translation': translated_text,
            'attention': attention_weights.tolist(),
            'source_tokens': source_tokens,
            'target_tokens': target_tokens
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
