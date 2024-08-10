import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Load a pre-trained model and tokenizer for detecting AI-generated text
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

def detect_ai_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # Extract probability of being AI-generated (label 1)
    ai_prob = probabilities[0][1].item()
    
    return ai_prob  # Returns the probability of being AI-generated

def get_color(probability):
    # Define color thresholds based on probability
    if probability > 0.8:
        return "red"  # Very likely AI-generated
    elif probability > 0.6:
        return "orange"  # Likely AI-generated
    elif probability > 0.4:
        return "yellow"  # Possibly AI-generated
    elif probability > 0.2:
        return "lightgreen"  # Unlikely AI-generated
    else:
        return "green"  # Very unlikely AI-generated

def analyze_text(input_text):
    # Split text into sentences or paragraphs for analysis
    sentences = input_text.split('\n')  # Splitting by paragraph. Use '. ' for sentence-level detection.
    
    highlighted_text = []
    for sentence in sentences:
        if sentence.strip():  # Skip empty lines
            ai_probability = detect_ai_text(sentence.strip())
            color = get_color(ai_probability)
            highlighted_text.append(f"<span style='background-color: {color}'>{sentence.strip()}</span>")
        else:
            highlighted_text.append("")  # Preserve paragraph breaks
    
    # Combine all parts into one HTML-formatted string
    result = "<br>".join(highlighted_text)
    
    # Add legend
    legend = (
        "<div style='margin-top: 20px;'>"
        "<strong>Legend:</strong><br>"
        "<span style='background-color: red'>Very likely AI-generated (80-100%)</span><br>"
        "<span style='background-color: orange'>Likely AI-generated (60-80%)</span><br>"
        "<span style='background-color: yellow'>Possibly AI-generated (40-60%)</span><br>"
        "<span style='background-color: lightgreen'>Unlikely AI-generated (20-40%)</span><br>"
        "<span style='background-color: green'>Very unlikely AI-generated (0-20%)</span><br>"
        "</div>"
    )
    
    return result + legend

# Gradio interface for direct text input and displaying results
iface = gr.Interface(
    fn=analyze_text, 
    inputs=gr.Textbox(lines=10, placeholder="Enter text here..."), 
    outputs="html",
    title="AI Text Detection Tool",
    description="Enter text to see which parts were written by AI. Different colors represent the model's certainty."
)

# Launch the interface
iface.launch()
