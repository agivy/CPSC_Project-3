# High-Quality Speech Transcription
# This script uses OpenAI's Whisper model for state-of-the-art speech recognition

import os
import time
import torch
import librosa
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import noisereduce as nr
import warnings
warnings.filterwarnings("ignore")

class HighQualityTranscriber:
    def __init__(self, model_size="large-v3"):
        """
        Initialize the transcriber with specified model size
        
        Args:
            model_size (str): Whisper model size - options:
                - "tiny": ~39M parameters
                - "base": ~74M parameters
                - "small": ~244M parameters
                - "medium": ~769M parameters
                - "large-v3": ~1.5B parameters (best quality)
        """
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load Whisper model and processor
        model_name = f"openai/whisper-{model_size}"
        print(f"Loading {model_name}...")
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"Model loaded successfully.")
        
    def preprocess_audio(self, audio_path, output_path=None, apply_noise_reduction=True, 
                        normalize_audio=True):
        """
        Preprocess audio for better transcription quality
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str, optional): Path to save processed audio
            apply_noise_reduction (bool): Whether to apply noise reduction
            normalize_audio (bool): Whether to normalize audio
            
        Returns:
            numpy.ndarray: Processed audio as numpy array
            int: Sample rate
        """
        print(f"Loading and preprocessing audio: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Apply preprocessing steps
        if apply_noise_reduction:
            print("Applying noise reduction...")
            # Estimate noise from a small section at the beginning
            noise_sample = audio[:int(0.5 * sr)]  # First 0.5 seconds
            audio = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=sr)
        
        if normalize_audio:
            print("Normalizing audio...")
            # Simple peak normalization to avoid using pyloudnorm
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save processed audio if output path is specified
        if output_path:
            print(f"Saving processed audio to: {output_path}")
            sf.write(output_path, audio, sr)
        
        return audio, sr
    
    def transcribe(self, audio, sample_rate=16000, chunk_length_s=30.0, 
              stride_length_s=5.0, language="english", return_timestamps=True):
        """
        Transcribe audio using Whisper model
        
        Args:
            audio (numpy.ndarray): Audio data as numpy array
            sample_rate (int): Audio sample rate
            chunk_length_s (float): Length of audio chunks in seconds
            stride_length_s (float): Stride length between chunks in seconds
            language (str): Language of the audio
            return_timestamps (bool): Whether to include word-level timestamps
            
        Returns:
            str: Full transcription text
            dict: Detailed results including segments with timestamps
        """
        print("Starting transcription...")
        start_time = time.time()
        
        # Calculate chunk and stride length in samples
        chunk_length = int(chunk_length_s * sample_rate)
        stride_length = int(stride_length_s * sample_rate)
        
        # For shorter audios, process the entire file at once
        if len(audio) <= chunk_length:
            # Create an attention mask (fixes the second warning)
            attention_mask = torch.ones_like(torch.tensor(audio)).unsqueeze(0)
            attention_mask = attention_mask[:, :len(audio)]
            
            inputs = self.processor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                return_attention_mask=True  # Request attention mask
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generation_config = self.model.generation_config
                generation_config.task = "transcribe"
                generation_config.language = language
                
                result = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_timestamps=return_timestamps
                )
                    
            transcription = self.processor.batch_decode(
                result, 
                skip_special_tokens=True, 
                normalize=False #preserve punctuation
            )[0]
            
            # Convert to detailed format
            detailed_results = {
                "text": transcription,
                "segments": [{
                    "text": transcription,
                    "start": 0,
                    "end": len(audio) / sample_rate
                }]
            }
            
        else:
            # Process longer audio in chunks with overlap
            all_segments = []
            full_text = []
            current_offset = 0
            
            for i in range(0, len(audio), stride_length):
                # Extract chunk
                chunk_end = min(i + chunk_length, len(audio))
                chunk = audio[i:chunk_end]
                
                # Skip chunks that are too short
                if len(chunk) < 0.5 * chunk_length:
                    continue
                
                chunk_duration = len(chunk) / sample_rate
                
                # Process chunk with attention mask
                inputs = self.processor(
                    chunk, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt",
                    return_attention_mask=True  # Request attention mask
                )
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    generation_config = self.model.generation_config
                    generation_config.task = "transcribe"
                    generation_config.language = language
                    
                    result = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        return_timestamps=return_timestamps
                    )
                
                transcription = self.processor.batch_decode(
                    result, 
                    skip_special_tokens=True, 
                    normalize=False #preserve punctuation
                )[0]
                
                # Add segment
                segment_info = {
                    "text": transcription,
                    "start": i / sample_rate,
                    "end": chunk_end / sample_rate
                }
                
                all_segments.append(segment_info)
                full_text.append(transcription)
                
                # Print progress
                progress = min(100, int((i + chunk_length) / len(audio) * 100))
                print(f"Transcription progress: {progress}%", end="\r")
                
                # Update offset for next chunk
                current_offset += chunk_duration
            
            # Merge overlapping segments
            detailed_results = {
                "text": " ".join(full_text),
                "segments": all_segments
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        audio_duration = len(audio) / sample_rate
        print(f"\nTranscription completed in {processing_time:.2f} seconds")
        print(f"Real-time factor: {processing_time/audio_duration:.2f}x")
        
        return detailed_results["text"], detailed_results
    
    def save_transcription(self, text, detailed_results, output_dir="transcriptions", 
                           base_filename="transcription"):
        """
        Save transcription results to files
        
        Args:
            text (str): Full transcription text
            detailed_results (dict): Detailed results including segments
            output_dir (str): Directory to save outputs
            base_filename (str): Base name for output files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plain text transcription
        text_file = os.path.join(output_dir, f"{base_filename}_{self.model_size}.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Save detailed JSON with segments and timestamps
        import json
        json_file = os.path.join(output_dir, f"{base_filename}_{self.model_size}_detailed.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Transcription saved to: {text_file}")
        print(f"Detailed results saved to: {json_file}")

def manually_combine_transcripts(transcript_files, output_file):
    """
    Combine multiple transcripts into a single file
    
    Args:
        transcript_files (list): List of transcript file paths
        output_file (str): Path to save combined transcript
    """
    combined_text = []
    
    for file_path in transcript_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            model_name = os.path.basename(file_path).split('_')[1].split('.')[0]
            combined_text.append(f"=== {model_name.upper()} MODEL TRANSCRIPT ===\n\n{content}\n\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_text))
    
    print(f"Combined transcript saved to: {output_file}")

def main():
    # Define model sizes to use - you can use multiple sizes or just one
    # For very high quality, use "large-v3" only if you have a good GPU
    # Otherwise, use "medium" which is still excellent
    model_sizes = ["medium", "large-v3"]
    
    # Input audio file
    audio_file = "audio_16khz.wav"
    
    # Create output directory
    output_dir = "transcriptions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process with each model size
    transcript_files = []
    
    for model_size in model_sizes:
        print(f"\n{'='*50}")
        print(f"Processing with {model_size} model")
        print(f"{'='*50}")
        
        try:
            # Initialize transcriber
            transcriber = HighQualityTranscriber(model_size=model_size)
            
            # Preprocess audio (with noise reduction and normalization)
            processed_audio_path = os.path.join(output_dir, f"processed_{os.path.basename(audio_file)}")
            audio, sr = transcriber.preprocess_audio(
                audio_file, 
                output_path=processed_audio_path,
                apply_noise_reduction=True,
                normalize_audio=True
            )
            
            # Transcribe audio with optimal settings for high quality
            # Reduced chunk size and stride for better accuracy
            text, detailed_results = transcriber.transcribe(
                audio, 
                sample_rate=sr,
                chunk_length_s=30.0,   # 30 seconds chunks (better for detailed transcription)
                stride_length_s=28.0,   # 28 seconds stride (2 seconds overlap)
                language="english",    # Specify language for better results
                return_timestamps=True # Include timestamps
            )
            
            # Save results
            file_path = os.path.join(output_dir, f"transcription_{model_size}_openai.txt")
            transcriber.save_transcription(
                text, 
                detailed_results, 
                output_dir=output_dir,
                base_filename="transcription"
            )
            
            transcript_files.append(file_path)
            
            # Free memory
            del transcriber
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing with {model_size} model: {e}")
    
    # # Combine all transcripts into a single file for easy comparison
    # if len(transcript_files) > 1:
    #     combined_file = os.path.join(output_dir, "combined_transcription.txt")
    #     manually_combine_transcripts(transcript_files, combined_file)
    
    print("\nAll transcriptions completed!")

if __name__ == "__main__":
    main()
