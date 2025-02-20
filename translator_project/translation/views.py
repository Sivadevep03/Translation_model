from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import json
import os
import numpy as np
import tensorflow as tf

# Initialize the model
model = None

def load_model():
    global model
    if model is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

@ensure_csrf_cookie
def index(request):
    # Try to load the model when the page is first loaded
    try:
        load_model()
    except Exception:
        pass
    return render(request, 'translation/index.html')

def preprocess_text(text):
    """
    Preprocess the input text according to your model's requirements.
    Modify this function based on how your model expects the input.
    """
    # Add your preprocessing steps here
    # For example:
    text = text.lower()
    text = ' '.join(text.split())
    return text

def postprocess_prediction(prediction):
    """
    Convert model output to readable text.
    Modify this function based on your model's output format.
    """
    # Add your postprocessing steps here
    # For example, if your model outputs indices:
    # return ' '.join([index_to_word.get(idx, '') for idx in prediction])
    return prediction

def translate_text(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            german_text = data.get('text', '')
            
            if not german_text:
                return JsonResponse({'error': 'No text provided'}, status=400)
            
            # Make sure model is loaded
            if model is None:
                load_model()
            
            try:
                # Preprocess the input
                processed_text = preprocess_text(german_text)
                
                # Make prediction
                # Note: Adjust this based on your model's input requirements
                prediction = model.predict([processed_text])
                
                # Post-process the prediction
                translated_text = postprocess_prediction(prediction)
                
                return JsonResponse({
                    'translation': translated_text
                })
                
            except Exception as e:
                return JsonResponse({'error': f'Translation error: {str(e)}'}, status=500)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON in request'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
