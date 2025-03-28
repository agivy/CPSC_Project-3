# CPSC_Project-3

The audio file is **"audio_16khz.wav"**.  

### Part I: Generating Transcriptions  
To generate transcriptions, run the following script:  

```bash
python part1_audio_transcript_openai.py
```

This will generate transcriptions in the folder: 
```bash
transcriptions/
```

The default transcription is: 
```bash
transcriptions/transcription_large-v3.txt
```

The original transcript is: 
```bash
transcriptions/transcription_actual.txt
```

### Part II: Fine-tuning & Defamation Analysis

First, create dataset using the script:

```bash
python dataset.py
```

This will save the dataset as 
```bash
defamation_dataset.json
```

Now, train an open-source LLM by running the following script:

```bash
python defamation-model-training.py
```

The script can be edited to train different LLMs.

The fine-tuned models will be saved in their respective folders named:

```bash
defamation-detector-model_{model_name}
```

Once fine-tuned, perform defamation analysis over the transcription by running:

```bash
python defamation-analysis.py
```

After executing, this script will produce detailed defamation results in the file:

```bash
defamation_analysis_results.json
```
The sample analysis report id for roberta-large model. For running different models, train the LLMs using the defamation-model-training.py script and modify the model_name field in defamation-analysis.py
