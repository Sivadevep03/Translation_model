# German-English Neural Machine Translation Web Interface

A web-based interface for German-to-English translation using neural machine translation with attention visualization.

## Features

- Real-time German to English translation
- Modern, responsive web interface
- Attention visualization for understanding the translation process
- Built with Django and PyTorch
- Uses the Helsinki-NLP/opus-mt-de-en model

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Django development server:
```bash
python manage.py migrate
python manage.py runserver
```

3. Open your browser and navigate to `http://localhost:8000`

## Usage

1. Enter German text in the input field
2. Click the "Translate" button
3. View the English translation and attention visualization below

## Technical Details

- Backend: Django
- Frontend: HTML, JavaScript, TailwindCSS
- Machine Translation: MarianMT model from Hugging Face
- Visualization: Plotly.js
