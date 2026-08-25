import asyncio
import json
import time
from uuid import uuid4
from vllm import AsyncLLMEngine, SamplingParams
import tqdm

class VLLMOAAS:
    def __init__(self):
        self.model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        # self.model_id = 'Qwen/Qwen3-32B'
        # self.model_id = "/Model"
        # self.model_id = 'microsoft/Phi-3.5-mini-instruct'
        self.model_id = '/nvmedata/hf_checkpoints/Qwen3-8B/'
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            gpu_memory_utilization=0.7,
            max_model_len=32768
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        with open('prompt.txt', 'r') as f:
            self.prompt_str = f.read()
        self.semaphore = asyncio.Semaphore(8)  # Limit concurrent requests

    #async def generate_response(self, input_json):
    async def async_run(self, input_data):
        #input_data = json.loads(input_json)
        title = input_data.get('title', '')
        max_tokens = input_data.get('max_tokens', 16)
        temperature = input_data.get('temperature', 0.3)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        user_prompt = self.prompt_str.replace("{Headline}", title)
        request_id = str(uuid4())

        #async with self.semaphore:
        #    final_output = None
        #    async for output in self.engine.generate(user_prompt, sampling_params, request_id):
        #        final_output = output
        #    return final_output.outputs[0].text if final_output else ""

        results_generator = self.engine.generate(user_prompt, sampling_params, request_id)
        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        assert final_output is not None
        generated_text = [final_output.outputs[0].text]
        # print(generated_text)
        return json.dumps(generated_text)

    async def run_batch_async(self, batch):
        tasks = [self.async_run(json.loads(line)) for line in batch]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    vllm_oaas = VLLMOAAS()
    runs = 16
    batch_size = 4
    loop = asyncio.get_event_loop()
    total_times = []

    with open('input.json') as f:
        lines = f.readlines()

    for i in tqdm.tqdm(range(runs)):
        vllm_oaas.engine.reset_prefix_cache()

        batch = []
        start_time = time.time()

        for line in lines:
            batch.append(line)
            if len(batch) == batch_size:
                try:
                    responses = loop.run_until_complete(
                        vllm_oaas.run_batch_async(batch)
                    )
                    # print(responses)
                except Exception as e:
                    print(f"Batch failed with error: {e}")
                batch = []

        if batch:
            try:
                loop.run_until_complete(
                    vllm_oaas.run_batch_async(batch)
                )
            except Exception as e:
                print(f"Final batch failed with error: {e}")

        end_time = time.time()
        duration = end_time - start_time
        total_times.append(duration)

    for i, duration in enumerate(total_times):
        qps = len(lines) / duration
        print(f"Run {i}: {duration:.2f}s, QPS: {qps:.2f}")

