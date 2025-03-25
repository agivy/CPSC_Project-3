# Analyze transcript for defamatory statements
# This script segments the transcript into sentences and classifies each one

import os
import torch
import pandas as pd
import numpy as np
import nltk
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize

# Download NLTK resources for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_transcript(file_path):
    """
    Load transcript from a text file
    
    Args:
        file_path (str): Path to the transcript file
        
    Returns:
        str: The transcript text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return None

def segment_transcript(transcript_text):
    """
    Segment transcript into sentences
    
    Args:
        transcript_text (str): The full transcript text
        
    Returns:
        list: List of sentences
    """
    # Clean the transcript (remove extra spaces, etc.)
    clean_text = ' '.join(transcript_text.split())
    
    # Split into sentences
    sentences = sent_tokenize(clean_text)
    
    # Additional cleaning for each sentence
    cleaned_sentences = []
    for sentence in sentences:
        # Skip very short sentences (likely not complete thoughts)
        if len(sentence.split()) < 3:
            continue
        cleaned_sentences.append(sentence.strip())
    
    return cleaned_sentences

def analyze_defamation(sentences, model_path):
    """
    Analyze sentences for defamatory content using the fine-tuned model
    
    Args:
        sentences (list): List of sentences to analyze
        model_path (str): Path to the fine-tuned model
        
    Returns:
        list: List of dictionaries with sentences and their classifications
    """
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    results = []
    
    # Process sentences in batches to avoid memory issues
    batch_size = 8
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # Tokenize the sentences
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)
            labels = torch.argmax(predictions, dim=1)
            confidence_scores = predictions.max(dim=1).values
        
        # Add results for this batch
        for j, sentence in enumerate(batch):
            is_defamatory = bool(labels[j].item())
            confidence = confidence_scores[j].item()
            
            results.append({
                "sentence": sentence,
                "is_defamatory": is_defamatory,
                "defamatory_bit": 1 if is_defamatory else 0,
                "confidence": confidence
            })
    
    return results

def save_results(results, output_file):
    """
    Save analysis results to a JSON file
    
    Args:
        results (list): Analysis results
        output_file (str): Path to save the results to
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    
    print(f"Analysis results saved to {output_file}")

def generate_summary(results):
    """
    Generate a summary of the defamation analysis
    
    Args:
        results (list): Analysis results
        
    Returns:
        dict: Summary statistics
    """
    total_sentences = len(results)
    defamatory_sentences = sum(1 for r in results if r["is_defamatory"])
    non_defamatory_sentences = total_sentences - defamatory_sentences
    
    defamatory_percentage = (defamatory_sentences / total_sentences) * 100 if total_sentences > 0 else 0
    
    # Find the sentence with highest defamation confidence
    if defamatory_sentences > 0:
        most_defamatory = max([r for r in results if r["is_defamatory"]], key=lambda x: x["confidence"])
    else:
        most_defamatory = None
    
    summary = {
        "total_sentences": total_sentences,
        "defamatory_sentences": defamatory_sentences,
        "non_defamatory_sentences": non_defamatory_sentences,
        "defamatory_percentage": defamatory_percentage,
        "most_defamatory_sentence": most_defamatory
    }
    
    return summary

def print_summary(summary):
    """
    Print a summary of the defamation analysis
    
    Args:
        summary (dict): Summary statistics
    """
    print("\n============= DEFAMATION ANALYSIS SUMMARY =============")
    print(f"Total sentences analyzed: {summary['total_sentences']}")
    print(f"Defamatory sentences: {summary['defamatory_sentences']} ({summary['defamatory_percentage']:.1f}%)")
    print(f"Non-defamatory sentences: {summary['non_defamatory_sentences']}")
    
    if summary['most_defamatory_sentence']:
        print("\nMost defamatory sentence:")
        print(f"  \"{summary['most_defamatory_sentence']['sentence']}\"")
        print(f"  Confidence: {summary['most_defamatory_sentence']['confidence']:.4f}")
    
    print("=========================================================\n")

def main():
    # Define path to the transcript and model
    transcript_file = "transcriptions/transcription_actual.txt"  # Adjust path if needed
    model_name = "google-bert/bert-large-cased"
    #other models:
    # google-bert/bert-base-cased
    # google-bert/bert-large-cased
    # distilbert-base-cased
    # roberta-large
    model_path = f"defamation-detector-model_{model_name}"
    output_file = f"defamation_analysis_results.json"
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run the fine-tuning script first.")
        return
    
    # Load the transcript
    transcript_text = load_transcript(transcript_file)
    if not transcript_text:
        print("Could not load transcript. Please check the file path.")
        return
    
    # Segment the transcript into sentences
    print("Segmenting transcript into sentences...")
    sentences = segment_transcript(transcript_text)
    print(f"Found {len(sentences)} sentences to analyze.")
    
    # Analyze each sentence for defamatory content
    print("Analyzing sentences for defamatory content...")
    results = analyze_defamation(sentences, model_path)
    
    # Save the results
    save_results(results, output_file)
    
    # Generate and print summary
    summary = generate_summary(results)
    print_summary(summary)
    
    # Create a defamation_bits file
    defamation_bits = [r["defamatory_bit"] for r in results]
    with open("defamation_bits.json", 'w') as f:
        json.dump(defamation_bits, f)
    print(f"Defamation bits saved to defamation_bits.json")

if __name__ == "__main__":
    main()
