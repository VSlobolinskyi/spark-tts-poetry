# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import asyncio
import json
import queue
import uuid
import functools
import aiohttp
import os
import time
import types
from pathlib import Path
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor


class UserData:
    """Track streaming data and processing statistics."""
    def __init__(self):
        self._completed_requests = queue.Queue()
        self._first_chunk_time = None
        self._start_time = None
        self._chunks = []
        self._errors = []

    def record_start_time(self):
        self._start_time = time.time()

    def add_chunk(self, chunk):
        if self._first_chunk_time is None:
            self._first_chunk_time = time.time()
        self._chunks.append(chunk)
        self._completed_requests.put(chunk)

    def add_error(self, error):
        self._errors.append(error)
        self._completed_requests.put(error)

    def get_first_chunk_latency(self):
        if self._first_chunk_time and self._start_time:
            return self._first_chunk_time - self._start_time
        return None

    def get_chunks(self):
        return self._chunks
    
    def has_errors(self):
        return len(self._errors) > 0
    
    def get_errors(self):
        return self._errors


def write_triton_stats(stats, summary_file):
    """Write Triton server statistics to a summary file."""
    with open(summary_file, "w") as summary_f:
        model_stats = stats["model_stats"]
        # write explanatory notes
        summary_f.write(
            "The log is parsing from triton server statistics, to better human readability. \n"
        )
        summary_f.write("To learn more about the log, please refer to: \n")
        summary_f.write("1. https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md \n")
        summary_f.write("2. https://github.com/triton-inference-server/server/issues/5374 \n\n")
        summary_f.write(
            "To better improve throughput, we always would like let requests wait in the queue for a while, and then execute them with a larger batch size. \n"
        )
        summary_f.write(
            "However, there is a trade-off between the increased queue time and the increased batch size. \n"
        )
        summary_f.write(
            "You may change 'max_queue_delay_microseconds' and 'preferred_batch_size' in the model configuration file to achieve this. \n"
        )
        summary_f.write(
            "See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#delayed-batching for more details. \n\n"
        )
        
        # write model statistics
        for model_state in model_stats:
            if "last_inference" not in model_state:
                continue
            summary_f.write(f"model name is {model_state['name']} \n")
            model_inference_stats = model_state["inference_stats"]
            total_queue_time_s = int(model_inference_stats["queue"]["ns"]) / 1e9
            total_infer_time_s = int(model_inference_stats["compute_infer"]["ns"]) / 1e9
            total_input_time_s = int(model_inference_stats["compute_input"]["ns"]) / 1e9
            total_output_time_s = int(model_inference_stats["compute_output"]["ns"]) / 1e9
            summary_f.write(
                f"queue time {total_queue_time_s:<5.2f} s, compute infer time {total_infer_time_s:<5.2f} s, compute input time {total_input_time_s:<5.2f} s, compute output time {total_output_time_s:<5.2f} s \n"
            )
            
            # write batch statistics
            model_batch_stats = model_state["batch_stats"]
            for batch in model_batch_stats:
                batch_size = int(batch["batch_size"])
                compute_input = batch["compute_input"]
                compute_output = batch["compute_output"]
                compute_infer = batch["compute_infer"]
                batch_count = int(compute_infer["count"])
                assert compute_infer["count"] == compute_output["count"] == compute_input["count"]
                compute_infer_time_ms = int(compute_infer["ns"]) / 1e6
                compute_input_time_ms = int(compute_input["ns"]) / 1e6
                compute_output_time_ms = int(compute_output["ns"]) / 1e6
                summary_f.write(
                    f"execuate inference with batch_size {batch_size:<2} total {batch_count:<5} times, total_infer_time {compute_infer_time_ms:<9.2f} ms, avg_infer_time {compute_infer_time_ms:<9.2f}/{batch_count:<5}={compute_infer_time_ms / batch_count:.2f} ms, avg_infer_time_per_sample {compute_infer_time_ms:<9.2f}/{batch_count:<5}/{batch_size}={compute_infer_time_ms / batch_count / batch_size:.2f} ms \n"
                )
                summary_f.write(
                    f"input {compute_input_time_ms:<9.2f} ms, avg {compute_input_time_ms / batch_count:.2f} ms, "
                )
                summary_f.write(
                    f"output {compute_output_time_ms:<9.2f} ms, avg {compute_output_time_ms / batch_count:.2f} ms \n"
                )


async def fetch_model_stats(server_url, model_name=""):
    """Fetch model statistics from Triton HTTP API."""
    url = f"{server_url}/v2/models{'/'+model_name if model_name else ''}/stats"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Failed to fetch statistics: {response.status}")
                    return None
    except Exception as e:
        print(f"Error fetching statistics: {e}")
        return None


async def fetch_model_config(server_url, model_name):
    """Fetch model configuration from Triton HTTP API."""
    url = f"{server_url}/v2/models/{model_name}/config"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Failed to fetch model config: {response.status}")
                    return None
    except Exception as e:
        print(f"Error fetching model config: {e}")
        return None


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="HTTP port of the triton server, default is 8000",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="",
        help="",
    )

    parser.add_argument(
        "--huggingface-dataset",
        type=str,
        default="yuekai/seed_tts",
        help="dataset name in huggingface dataset hub",
    )

    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="dataset split name, default is 'test'",
    )

    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to the manifest dir which includes wav.scp trans.txt files.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="f5_tts",
        choices=["f5_tts", "spark_tts"],
        help="triton model_repo module name to request: transducer for k2, attention_rescoring for wenet offline, streaming_wenet for wenet streaming, infer_pipeline for paraformer large offline",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of concurrent tasks for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default="./tmp",
        help="log directory",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "streaming"],
        help="Select offline or streaming benchmark mode."
    )
    
    parser.add_argument(
        "--chunk-overlap-duration",
        type=float,
        default=0.1,
        help="Chunk overlap duration for streaming reconstruction (in seconds)."
    )

    return parser.parse_args()


def load_audio(wav_path, target_sample_rate=16000):
    """Load audio file and resample if necessary."""
    assert target_sample_rate == 16000, "hard coding in server"
    if isinstance(wav_path, dict):
        waveform = wav_path["array"]
        sample_rate = wav_path["sampling_rate"]
    else:
        waveform, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample

        num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
        waveform = resample(waveform, num_samples)
    return waveform, target_sample_rate


def prepare_request_input_output(
    waveform,
    reference_text,
    target_text,
    sample_rate=16000,
    padding_duration: int = None
):
    """Prepares inputs for Triton HTTP inference (offline or streaming)."""
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    # Apply padding only if padding_duration is provided (for offline)
    if padding_duration:
        duration = len(waveform) / sample_rate
        # Estimate target duration based on text length ratio (crude estimation)
        # Avoid division by zero if reference_text is empty
        if reference_text:
             estimated_target_duration = duration / len(reference_text) * len(target_text)
        else:
             estimated_target_duration = duration # Assume target duration similar to reference if no text

        # Calculate required samples based on estimated total duration
        required_total_samples = padding_duration * sample_rate * (
            (int(estimated_target_duration + duration) // padding_duration) + 1
        )
        samples = np.zeros((1, required_total_samples), dtype=np.float32)
        samples[0, : len(waveform)] = waveform
    else:
        # No padding for streaming or if padding_duration is None
        samples = waveform.reshape(1, -1).astype(np.float32)
    
    # Prepare HTTP request data
    data = {
        "inputs": [
            {
                "name": "reference_wav",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist()
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
                "data": [reference_text]
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text]
            }
        ]
    }

    return data


async def run_streaming_inference(
    session,
    server_url,
    model_name,
    request_data,
    request_id,
    user_data,
    chunk_overlap_duration,
    save_sample_rate,
    audio_save_path,
):
    """Execute HTTP streaming inference request."""
    start_time_total = time.time()
    user_data.record_start_time()
    
    url = f"{server_url}/v2/models/{model_name}/infer"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",  # SSE for streaming
        "Inference-Header-Content-Length": str(len(json.dumps(request_data)))
    }
    
    # Add the query parameters for streaming
    params = {
        "request_id": request_id,
        "stream": "true"
    }
    
    try:
        async with session.post(url, headers=headers, json=request_data, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                error = Exception(f"Streaming request failed with status {response.status}: {error_text}")
                user_data.add_error(error)
                return None, None, None
            
            # Process the SSE stream
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data:'):
                    try:
                        chunk_data = json.loads(line[5:])  # Remove 'data:' prefix
                        
                        # Extract audio chunk from response
                        if 'outputs' in chunk_data and len(chunk_data['outputs']) > 0:
                            audio_data = chunk_data['outputs'][0]['data']
                            audio_chunk = np.array(audio_data, dtype=np.float32)
                            
                            if audio_chunk.size > 0:
                                user_data.add_chunk(audio_chunk)
                        
                        # Check if this is the final response
                        if 'parameters' in chunk_data and 'triton_final_response' in chunk_data['parameters']:
                            if chunk_data['parameters']['triton_final_response'].get('bool_param', False):
                                break
                    except json.JSONDecodeError as e:
                        print(f"Error parsing SSE data: {e}, line: {line}")
                        continue
    
    except Exception as e:
        user_data.add_error(e)
        print(f"Error during streaming request: {e}")
        return None, None, None
    
    end_time_total = time.time()
    total_request_latency = end_time_total - start_time_total
    first_chunk_latency = user_data.get_first_chunk_latency()
    
    # Process all received chunks
    audios = user_data.get_chunks()
    
    # Reconstruct audio using cross-fade
    actual_duration = 0
    if audios:
        cross_fade_samples = int(chunk_overlap_duration * save_sample_rate)
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)
        
        if len(audios) == 1:
            reconstructed_audio = audios[0]
        else:
            reconstructed_audio = audios[0][:-cross_fade_samples]  # Start with first chunk minus overlap
            for i in range(1, len(audios)):
                # Cross-fade section
                cross_faded_overlap = (
                    audios[i][:cross_fade_samples] * fade_in +
                    audios[i-1][-cross_fade_samples:] * fade_out
                )
                
                # Middle section of the current chunk
                if len(audios[i]) > 2 * cross_fade_samples:
                    middle_part = audios[i][cross_fade_samples:-cross_fade_samples]
                    # Concatenate
                    reconstructed_audio = np.concatenate([reconstructed_audio, cross_faded_overlap, middle_part])
                else:
                    # Handle case where chunk is too small for full cross-fade
                    reconstructed_audio = np.concatenate([reconstructed_audio, cross_faded_overlap])
            
            # Add the last part of the final chunk
            if len(audios[-1]) > cross_fade_samples:
                reconstructed_audio = np.concatenate([reconstructed_audio, audios[-1][-cross_fade_samples:]])
        
        # Calculate actual duration and save audio
        actual_duration = len(reconstructed_audio) / save_sample_rate
        os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)
        sf.write(audio_save_path, reconstructed_audio, save_sample_rate, "PCM_16")
    else:
        print("Warning: No audio chunks received.")
        actual_duration = 0
    
    return total_request_latency, first_chunk_latency, actual_duration


async def send_streaming(
    manifest_item_list: list,
    name: str,
    server_url: str,
    log_interval: int,
    model_name: str,
    audio_save_dir: str = "./",
    save_sample_rate: int = 16000,
    chunk_overlap_duration: float = 0.1,
    padding_duration: int = None,
):
    """Process items with streaming inference."""
    total_duration = 0.0
    latency_data = []
    task_id = int(name[5:])
    
    # Use aiohttp for HTTP requests
    async with aiohttp.ClientSession() as session:
        print(f"{name}: Starting streaming processing for {len(manifest_item_list)} items.")
        for i, item in enumerate(manifest_item_list):
            if i % log_interval == 0:
                print(f"{name}: Processing item {i}/{len(manifest_item_list)}")

            try:
                waveform, sample_rate = load_audio(item["audio_filepath"], target_sample_rate=16000)
                reference_text, target_text = item["reference_text"], item["target_text"]

                request_data = prepare_request_input_output(
                    waveform,
                    reference_text,
                    target_text,
                    sample_rate,
                    padding_duration=padding_duration
                )
                
                request_id = str(uuid.uuid4())
                user_data = UserData()
                audio_save_path = os.path.join(audio_save_dir, f"{item['target_audio_path']}.wav")

                total_request_latency, first_chunk_latency, actual_duration = await run_streaming_inference(
                    session,
                    server_url,
                    model_name,
                    request_data,
                    request_id,
                    user_data,
                    chunk_overlap_duration,
                    save_sample_rate,
                    audio_save_path
                )

                if total_request_latency is not None:
                    print(f"{name}: Item {i} - First Chunk Latency: {first_chunk_latency:.4f}s, Total Latency: {total_request_latency:.4f}s, Duration: {actual_duration:.4f}s")
                    latency_data.append((total_request_latency, first_chunk_latency, actual_duration))
                    total_duration += actual_duration
                else:
                    print(f"{name}: Item {i} failed.")

            except FileNotFoundError:
                print(f"Error: Audio file not found for item {i}: {item['audio_filepath']}")
            except Exception as e:
                print(f"Error processing item {i} ({item['target_audio_path']}): {e}")
                import traceback
                traceback.print_exc()

    print(f"{name}: Finished streaming processing. Total duration synthesized: {total_duration:.4f}s")
    return total_duration, latency_data


async def send(
    manifest_item_list: list,
    name: str,
    server_url: str,
    log_interval: int,
    model_name: str,
    padding_duration: int = None,
    audio_save_dir: str = "./",
    save_sample_rate: int = 16000,
):
    """Execute HTTP offline (non-streaming) inference requests."""
    total_duration = 0.0
    latency_data = []
    task_id = int(name[5:])
    
    # Use aiohttp for async HTTP requests
    async with aiohttp.ClientSession() as session:
        url = f"{server_url}/v2/models/{model_name}/infer"
        headers = {"Content-Type": "application/json"}
        
        for i, item in enumerate(manifest_item_list):
            if i % log_interval == 0:
                print(f"{name}: {i}/{len(manifest_item_list)}")
                
            waveform, sample_rate = load_audio(item["audio_filepath"], target_sample_rate=16000)
            reference_text, target_text = item["reference_text"], item["target_text"]
            
            request_data = prepare_request_input_output(
                waveform,
                reference_text,
                target_text,
                sample_rate,
                padding_duration=padding_duration
            )
            
            sequence_id = 100000000 + i + task_id * 10
            start = time.time()
            
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=request_data,
                    params={"request_id": str(sequence_id)}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Request failed with status {response.status}: {error_text}")
                        continue
                        
                    result = await response.json()
                    audio = np.array(result["outputs"][0]["data"], dtype=np.float32)
                    actual_duration = len(audio) / save_sample_rate
                    
                    end = time.time() - start
                    
                    audio_save_path = os.path.join(audio_save_dir, f"{item['target_audio_path']}.wav")
                    os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)
                    sf.write(audio_save_path, audio, save_sample_rate, "PCM_16")
                    
                    latency_data.append((end, actual_duration))
                    total_duration += actual_duration
                    
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                import traceback
                traceback.print_exc()
    
    return total_duration, latency_data


def load_manifests(manifest_path):
    """Load manifests from a file."""
    with open(manifest_path, "r") as f:
        manifest_list = []
        for line in f:
            assert len(line.strip().split("|")) == 4
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            utt = Path(utt).stem
            # gt_wav = os.path.join(os.path.dirname(manifest_path), "wavs", utt + ".wav")
            if not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(os.path.dirname(manifest_path), prompt_wav)
            manifest_list.append(
                {
                    "audio_filepath": prompt_wav,
                    "reference_text": prompt_text,
                    "target_text": gt_text,
                    "target_audio_path": utt,
                }
            )
    return manifest_list


def split_data(data, k):
    """Split data into k parts."""
    n = len(data)
    if n < k:
        print(f"Warning: the length of the input list ({n}) is less than k ({k}). Setting k to {n}.")
        k = n

    quotient = n // k
    remainder = n % k

    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient

        result.append(data[start:end])
        start = end

    return result


async def main():
    """Main function to execute inference requests."""
    args = get_args()
    server_url = f"http://{args.server_addr}:{args.server_port}"
    
    if args.reference_audio:
        args.num_tasks = 1
        args.log_interval = 1
        manifest_item_list = [
            {
                "reference_text": args.reference_text,
                "target_text": args.target_text,
                "audio_filepath": args.reference_audio,
                "target_audio_path": "test",
            }
        ]
    elif args.huggingface_dataset:
        import datasets

        dataset = datasets.load_dataset(
            args.huggingface_dataset,
            split=args.split_name,
            trust_remote_code=True,
        )
        manifest_item_list = []
        for i in range(len(dataset)):
            manifest_item_list.append(
                {
                    "audio_filepath": dataset[i]["prompt_audio"],
                    "reference_text": dataset[i]["prompt_text"],
                    "target_audio_path": dataset[i]["id"],
                    "target_text": dataset[i]["target_text"],
                }
            )
    else:
        manifest_item_list = load_manifests(args.manifest_path)

    num_tasks = min(args.num_tasks, len(manifest_item_list))
    manifest_item_list = split_data(manifest_item_list, num_tasks)

    os.makedirs(args.log_dir, exist_ok=True)
    tasks = []
    start_time = time.time()
    
    for i in range(num_tasks):
        # Task creation based on mode
        if args.mode == "offline":
            task = asyncio.create_task(
                send(
                    manifest_item_list[i],
                    name=f"task-{i}",
                    server_url=server_url,
                    log_interval=args.log_interval,
                    model_name=args.model_name,
                    audio_save_dir=args.log_dir,
                    padding_duration=1,
                    save_sample_rate=24000 if args.model_name == "f5_tts" else 16000,
                )
            )
        elif args.mode == "streaming":
            task = asyncio.create_task(
                send_streaming(
                    manifest_item_list[i],
                    name=f"task-{i}",
                    server_url=server_url,
                    log_interval=args.log_interval,
                    model_name=args.model_name,
                    audio_save_dir=args.log_dir,
                    padding_duration=10,
                    save_sample_rate=24000 if args.model_name == "f5_tts" else 16000,
                    chunk_overlap_duration=args.chunk_overlap_duration,
                )
            )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        if ans:
            total_duration += ans[0]
            latency_data.extend(ans[1])
        else:
            print("Warning: A task returned None, possibly due to an error.")

    if total_duration == 0:
        print("Total synthesized duration is zero. Cannot calculate RTF or latency percentiles.")
        rtf = float('inf')
    else:
        rtf = elapsed / total_duration

    s = f"Mode: {args.mode}\n"
    s += f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration / 3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds ({elapsed / 3600:.2f} hours)\n"

    # Statistics reporting based on mode
    if latency_data:
        if args.mode == "offline":
            # Original offline latency calculation
            latency_list = [chunk_end for (chunk_end, chunk_duration) in latency_data]
            if latency_list:
                latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
                latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
                s += f"latency_variance: {latency_variance:.2f}\n"
                s += f"latency_50_percentile_ms: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
                s += f"latency_90_percentile_ms: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
                s += f"latency_95_percentile_ms: {np.percentile(latency_list, 95) * 1000.0:.2f}\n"
                s += f"latency_99_percentile_ms: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
                s += f"average_latency_ms: {latency_ms:.2f}\n"
            else:
                s += "No latency data collected for offline mode.\n"

        elif args.mode == "streaming":
            # Calculate stats for total request latency and first chunk latency
            total_latency_list = [total for (total, first, duration) in latency_data if total is not None]
            first_chunk_latency_list = [first for (total, first, duration) in latency_data if first is not None]

            s += "\n--- Total Request Latency ---\n"
            if total_latency_list:
                avg_total_latency_ms = sum(total_latency_list) / len(total_latency_list) * 1000.0
                variance_total_latency = np.var(total_latency_list, dtype=np.float64) * 1000.0
                s += f"total_request_latency_variance: {variance_total_latency:.2f}\n"
                s += f"total_request_latency_50_percentile_ms: {np.percentile(total_latency_list, 50) * 1000.0:.2f}\n"
                s += f"total_request_latency_90_percentile_ms: {np.percentile(total_latency_list, 90) * 1000.0:.2f}\n"
                s += f"total_request_latency_95_percentile_ms: {np.percentile(total_latency_list, 95) * 1000.0:.2f}\n"
                s += f"total_request_latency_99_percentile_ms: {np.percentile(total_latency_list, 99) * 1000.0:.2f}\n"
                s += f"average_total_request_latency_ms: {avg_total_latency_ms:.2f}\n"
            else:
                s += "No total request latency data collected.\n"

            s += "\n--- First Chunk Latency ---\n"
            if first_chunk_latency_list:
                avg_first_chunk_latency_ms = sum(first_chunk_latency_list) / len(first_chunk_latency_list) * 1000.0
                variance_first_chunk_latency = np.var(first_chunk_latency_list, dtype=np.float64) * 1000.0
                s += f"first_chunk_latency_variance: {variance_first_chunk_latency:.2f}\n"
                s += f"first_chunk_latency_50_percentile_ms: {np.percentile(first_chunk_latency_list, 50) * 1000.0:.2f}\n"
                s += f"first_chunk_latency_90_percentile_ms: {np.percentile(first_chunk_latency_list, 90) * 1000.0:.2f}\n"
                s += f"first_chunk_latency_95_percentile_ms: {np.percentile(first_chunk_latency_list, 95) * 1000.0:.2f}\n"
                s += f"first_chunk_latency_99_percentile_ms: {np.percentile(first_chunk_latency_list, 99) * 1000.0:.2f}\n"
                s += f"average_first_chunk_latency_ms: {avg_first_chunk_latency_ms:.2f}\n"
            else:
                s += "No first chunk latency data collected.\n"
    else:
        s += "No latency data collected.\n"

    print(s)
    if args.manifest_path:
        name = Path(args.manifest_path).stem
    elif args.split_name:
        name = args.split_name
    elif args.reference_audio:
        name = Path(args.reference_audio).stem
    else:
        name = "results"  # Default name if no manifest/split/audio provided
    with open(f"{args.log_dir}/rtf-{name}.txt", "w") as f:
        f.write(s)

    # Fetch and write server statistics
    try:
        stats = await fetch_model_stats(server_url)
        if stats:
            write_triton_stats(stats, f"{args.log_dir}/stats_summary-{name}.txt")
        
        metadata = await fetch_model_config(server_url, args.model_name)
        if metadata:
            with open(f"{args.log_dir}/model_config-{name}.json", "w") as f:
                json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Could not retrieve statistics or config: {e}")


if __name__ == "__main__":
    async def run_main():
        try:
            await main()
        except Exception as e:
            print(f"An error occurred in main: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_main())