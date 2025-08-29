import numpy as np
import requests
import soundfile as sf
import torch
import json
from pathlib import Path

class TritonSparkTTS:
    """Client for Spark TTS using Triton Inference Server"""
    
    def __init__(self, server_url="localhost:8000", model_name="spark_tts"):
        """Initialize the Triton client"""
        self.server_url = f"http://{server_url}"
        self.model_name = model_name
        self.sample_rate = 16000  # Default sample rate
        
        # Check server health
        try:
            response = requests.get(f"{self.server_url}/v2/health/ready")
            if response.status_code != 200:
                print(f"Warning: Triton server at {server_url} may not be ready: {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to Triton server at {server_url}: {e}")
        
    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path=None,
        prompt_text=None,
        gender=None,
        pitch=None,
        speed=None,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
    ):
        """
        Perform inference via Triton server
        
        Args match the original SparkTTS.inference() method
        """
        if prompt_speech_path is None and gender is not None:
            raise NotImplementedError(
                "Voice creation mode not yet implemented for Triton client"
            )
            
        if prompt_speech_path is None:
            raise ValueError("Reference audio (prompt_speech_path) is required")
            
        # Load reference audio
        waveform, sample_rate = sf.read(prompt_speech_path)
        
        # Prepare inputs for Triton request
        waveform = waveform.reshape(1, -1).astype(np.float32)
        lengths = np.array([[len(waveform[0])]], dtype=np.int32)
        
        # Create input data
        request_data = {
            "inputs": [
                {
                    "name": "reference_wav",
                    "shape": waveform.shape,
                    "datatype": "FP32",
                    "data": waveform.tolist()
                },
                {
                    "name": "reference_wav_len",
                    "shape": lengths.shape,
                    "datatype": "INT32",
                    "data": lengths.tolist()
                },
                {
                    "name": "reference_text",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [prompt_text if prompt_text else ""]
                },
                {
                    "name": "target_text",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [text]
                }
            ]
        }
        
        # Send request to Triton server
        url = f"{self.server_url}/v2/models/{self.model_name}/infer"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, headers=headers, json=request_data)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Parse response
            result = response.json()
            audio = np.array(result["outputs"][0]["data"], dtype=np.float32)
            
            # Return as torch tensor to match original API
            return torch.tensor(audio)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            # Return empty audio on error
            return torch.tensor([0.0], dtype=torch.float32)