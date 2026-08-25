import io
import os
import sys
import json
import torch
import time
import csv
import threading
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, GenerationConfig
from vllm import LLM, SamplingParams
import base64
from PIL import Image
import requests
from io import BytesIO
import tqdm
import socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = (
    HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])


class VLLMOAAS:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        # self.model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        # self.model_id = 'microsoft/Phi-3.5-mini-instruct'
        self.model_id = '/nvmedata/hf_checkpoints/Qwen3-8B/'
        self.engine = LLM(model=self.model_id, 
                          trust_remote_code=True,
                          enable_prefix_caching=True, 
                          enable_chunked_prefill=True, 
                          tensor_parallel_size=1,
                          dtype="float16")
        #self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.prompt_str = open('prompt.txt', 'r').read()
    
    def run(self, input):        
        inputJson = json.loads(input)
        # print(inputJson)
        title = inputJson.get('title')
        if 'max_tokens' in inputJson:
            max_tokens = inputJson.get('max_tokens')
        else:
            max_tokens = 16
        if 'temperature' in inputJson:
            temperature = inputJson.get('temperature')
        else:
            temperature = 0.3
        sampling_params = SamplingParams(
            max_tokens = max_tokens, temperature = temperature)
        user_prompt = self.prompt_str.replace("{Headline}", title)
        messages =[{ "role": "user", "content": user_prompt}]
        #text = self.tokenizer.apply_chat_template(messages)
        outputs = self.engine.generate(user_prompt, sampling_params, use_tqdm=False)

        res = []
        for o in outputs:
            generated_text = o.outputs[0].text
            # print(generated_text)
            res.append(generated_text)

        return json.dumps(res)

    def run_demo(self):
        title = "Hello Microsoft"
        prompt = self.prompt_str.replace("{Headline}", title)
        sampling_params = SamplingParams(max_tokens=16, temperature=0.3)
        #print(text)
        outputs = self.engine.generate(prompt, sampling_params, )
        # print(outputs)
        for o in outputs:
            generated_text = o.outputs[0].text
            # print(generated_text)


if __name__ == "__main__":
    vllm_oaas = VLLMOAAS()
    if len(sys.argv) > 1:
        input = sys.argv[1]
    #print(vllm_oaas.run_demo())
    runs = 16
    total_times = []
    for i in tqdm.tqdm(range(runs)):
        vllm_oaas.engine.reset_prefix_cache()
        with open('input.json') as f:
            lines = f.readlines()            
            tm0 = time.time()
            for line in lines:            
                # print(line)
                # print(vllm_oaas.run(line))
                resp = vllm_oaas.run(line)
                # print(resp)
            tm1 = time.time()
            total_times.append(tm1 - tm0)

    for i in range(len(total_times)):
        print(f"Run {i}: {total_times[i]}")
        # Calculate the QPS
        qps = len(lines) / total_times[i]
        print(f"QPS: {qps}")