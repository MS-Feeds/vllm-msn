#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Analyzer Module for Content Recommender using vLLM OAAS Async Engine

This module handles the analysis of user browsing/search history 
to infer user interests using vLLM's Async Engine for improved performance.

Author: Chen
Date: May 31, 2025
"""
import io
import os
import sys
import json
import torch
import time
import csv
import asyncio
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
import json
import logging
import time
import datetime
from typing import List, Dict, Any, Callable, AsyncGenerator
import pandas as pd
import tqdm
import torch
from uuid import uuid4
import argparse
import pathlib

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Set vLLM logger level
from vllm.logger import init_logger
vllm_logger = init_logger(__name__)
vllm_logger.setLevel(logging.ERROR)


def prepare_history_analysis_prompt(user_profile: Dict) -> Dict:
    """
    Prepare a prompt for the LLM to analyze user profile and infer interests.

    Args:
        user_profile: Dictionary containing user profile data including interests

    Returns:
        Dict: Formatted prompt for the model
    """
    # Get current date information for timestamp context
    current_date = datetime.datetime.now()
    current_year = current_date.strftime("%Y")
    current_month = current_date.strftime("%B")  # Full month name

    # Extract user profile information
    age = user_profile.get('Age', 'Unknown')
    gender = user_profile.get('Gender', 'Unknown')
    interests = user_profile.get('Interests', [])

    # Add demographic context to the prompt
    demographic_context = f"Age: {age}\nGender: {gender}\n\n"

    # Create interests list string with categories and keywords
    interests_text = []
    if interests:
        for interest in interests:
            category = interest.get('category', '')
            keywords = interest.get('keywords', [])
            if category and keywords:
                keywords_str = ", ".join(keywords)
                interests_text.append(f"{category}: {keywords_str}")

    interests_list = "\n".join(
        interests_text) if interests_text else "No explicit interests provided"

    # Build the prompt
    prompt_content = f"""Analyze the following user's profile to identify the most valuable niche interests:

{demographic_context}
Summarized topics from user browsing and search history (order by recency and strength of interest):
{interests_list}

Your task:
- Identify 1-5 high-confidence, distinct, and valuable niche interests worth tracking over time.
- Only include interests if you are highly confident, based on clear evidence in the profile.
- Only include interests likely to have fresh, relevant content in the near future.
- Ensure all interests are distinct and diversified (no overlap).
- Consider Age and Gender if available, as these may influence interest relevance.
- Use the current time ({current_month} {current_year}) as context; avoid outdated or off-season interests.

For each inferred interest generate a consistent package:
- name: Concise, relevant title (≤10 words)
- description: Detailed explanation of what tracking this interest entails
- keywords: Core entities/concepts/aspects
- queries: 5–10 concise (≤60 chars), English (en-US) search queries relevant to the interest.
- reason: Brief explanation for the recommendation
- strength: Confidence rating (1–10)
Make sure name, description, keywords, reason and queries are aligned with each other.

Guidelines for queries:
- Queries will be used with the Bing News API, which is keyword-based (not semantic search). So avoid generic, vague, or redundant terms; ensure each query is specific and relevant to the interest. No need to include unnecessary freshness or date-related words (e.g., "latest", "today", "news").
- Only generate queries for interests that are likely to have fresh, relevant news content in the near future.
- Ensure all queries for an interest are unique, diverse and collectively cover a wide range of subtopics, perspectives, and facets related to the interest.

Return the result as JSON in this format:
[
  {{
    "name": "Interest Name",
    "description": "Detailed description",
    "keywords": ["keyword1", "keyword2"],
    "queries": ["Query 1", "Query 2"],
    "reason": "Why this interest is recommended",
    "strength": 7
  }}
]

If no high-confidence interests can be determined, return "can't infer".
"""
    return {
        "messages": [
            {
                "role": "system",
                "content": """You are an advanced AI model specializing in analyzing user profiles to infer high-confidence, diverse, and valuable niche interests for ongoing tracking and content recommendation.

Your recommendations must:
- Be based on clear evidence from the user's profile
- Have ongoing relevance, with new content regularly published
- Be uniquely tailored to the user's actual interests (not generic topics)
- Enable highly personalized content recommendations

When analyzing user profile:
1. Prioritize recent and repeated activities
2. Identify patterns indicating sustained or growing interest
3. Focus on specific niches, not broad categories
4. Recommend topics where tracking future developments will benefit the user
5. Ensure each recommended interest is meaningfully different from the others

Do NOT recommend:
- Adult, offensive, or crime-related topics
- One-off or transient interests
- Topics unlikely to have fresh future content
- Generic, non-personalized categories
- Low-confidence inferences based on limited evidence"""
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ]
    }


class VLLMOAAS:
    def __init__(self, model_path: str = "/nvmedata/hf_checkpoints/Qwen3-32B/"):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        # Initialize Async Engine
        engine_args = AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=2,
            dtype="bfloat16",
            gpu_memory_utilization=0.95,
            max_num_batched_tokens=16384,
            enable_chunked_prefill=True,
            max_num_seqs=512,
            swap_space=8,
            disable_log_stats=True,
            disable_log_requests=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
            stop=None,
        )

        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(64)

    async def __call__(self, messages: Dict) -> Dict:
        """
        Make the class callable to handle model inference asynchronously.

        Args:
            messages: Dictionary containing the messages for the model

        Returns:
            Dict: Model response containing the output
        """
        assert isinstance(messages, dict), "Messages must be a dictionary"
        assert "messages" in messages, "Messages must contain a 'messages' key"
        assert isinstance(
            messages["messages"], list), "Messages must contain a list of messages"
        assert all(isinstance(msg, dict)
                   for msg in messages["messages"]), "Messages must contain a list of dictionaries"
        assert all(
            "role" in msg and "content" in msg for msg in messages["messages"]), "Messages must contain a 'role' and 'content' key"
        assert all(msg["role"] in ["system", "user"] for msg in messages["messages"]
                   ), "Messages must contain a 'role' of 'system' or 'user'"
        assert all(isinstance(msg["content"], str) for msg in messages["messages"]
                   ), "Messages must contain a 'content' of type string"

        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # Generate unique request ID
            request_id = str(uuid4())

            # Use semaphore to limit concurrent requests
            async with self.semaphore:
                # Generate response asynchronously
                final_output = None
                async for request_output in self.engine.generate(prompt, self.sampling_params, request_id):
                    final_output = request_output

                if final_output is None:
                    raise Exception("No output generated")

                response = final_output.outputs[0].text

                # Format response to match the expected structure
                return {
                    "choices": [{
                        "message": {
                            "content": response
                        }
                    }]
                }
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise


async def analyze_user_history(user_profile: Dict, llm_caller: Callable) -> List[Dict]:
    """
    Analyze user profile to infer interests using LLM asynchronously.

    Args:
        user_profile: Dictionary containing user profile data
        llm_caller: Async function to call LLM

    Returns:
        List[Dict]: Inferred user interests
    """
    prompt = prepare_history_analysis_prompt(user_profile)
    response = await llm_caller(prompt)

    try:
        assert 'choices' in response, "Invalid response format: missing 'choices'"
        if "choices" not in response or not response["choices"]:
            logger.error("Invalid response format: missing 'choices'")
            return []

        content = response["choices"][0]["message"]["content"]
        logger.info(f"Raw LLM response: {content[:200]}...")
        
        if "can't infer" in content.lower():
            logger.warning(
                "LLM couldn't infer any interests from the provided history")
            return []

        # Extract JSON from the content
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        if not json_str or (not json_str.startswith("[") and not json_str.endswith("]")):
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]

        try:
            inferred_interests = json.loads(json_str)
        except json.JSONDecodeError as json_err:
            logger.error(
                f"JSON decode error: {json_err}. Attempting to clean and retry...")
            # logger.error(f"Raw JSON string: {json_str}")    
            cleaned_json_str = json_str.replace("'", '"').replace("\\", "\\\\")
            inferred_interests = json.loads(cleaned_json_str)

        if not isinstance(inferred_interests, list):
            if isinstance(inferred_interests, dict):
                inferred_interests = [inferred_interests]
            else:
                return []

        validated_interests = []
        for interest in inferred_interests:
            if "name" in interest and "reason" in interest:
                if "queries" not in interest or not interest["queries"]:
                    interest["queries"] = [f"latest {interest['name']} news"]

                if not isinstance(interest["queries"], list):
                    interest["queries"] = [interest["queries"]]

                if "description" not in interest or not interest["description"]:
                    interest["description"] = interest["reason"]

                if "keywords" not in interest or not interest["keywords"]:
                    interest["keywords"] = interest["name"].split()

                if "strength" not in interest or not isinstance(interest["strength"], (int, float)):
                    interest["strength"] = 5
                else:
                    interest["strength"] = min(
                        max(interest["strength"], 1), 10)

                validated_interests.append(interest)

        return validated_interests
    except Exception as e:
        logger.error(f"Error parsing inferred interests: {e}")
        return []


def parse_raw_user_data(raw_data: str) -> Dict:
    """Parse raw user data string into a structured profile dictionary."""
    user_profile = {}

    fields = raw_data.split('|')
    for field in fields:
        if ':' not in field:
            continue
        key, value = field.split(':', 1)
        key = key.strip()
        value = value.strip()

        if key == "Age":
            user_profile["Age"] = value
        elif key == "Gender":
            user_profile["Gender"] = value
        elif key == "GPTQueryHistory":
            interests = []
            categories = value.split(';')
            for category_entry in categories:
                if ':' not in category_entry:
                    continue
                category, keywords_str = category_entry.split(':', 1)
                keywords = [kw.strip()
                            for kw in keywords_str.split(',') if kw.strip()]
                interests.append({
                    "category": category.strip(),
                    "keywords": keywords
                })
            user_profile["Interests"] = interests

    return user_profile


async def process_batch(batch: List[Dict], llm_caller: Callable) -> List[List[Dict]]:
    """Process a batch of user profiles concurrently."""
    tasks = [analyze_user_history(user_profile, llm_caller)
             for user_profile in batch]
    return await asyncio.gather(*tasks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/genz_users_20k_format.tsv")
    parser.add_argument("--output_path", type=str, default="../output/genz_users_interests_vllm_oaas_async.jsonl")
    parser.add_argument("--model_path", type=str, default="../models/Qwen3-8B")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


async def main():
    args = parse_args()
    df = pd.read_csv(args.input_path, sep='\t')
    llm_caller = VLLMOAAS(model_path=args.model_path)

    # Process users in batches
    batch_size = args.batch_size  # Adjust based on your GPU memory and performance needs
    start_time = time.time()

    for i in tqdm.tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        user_profiles = [parse_raw_user_data(
            user['profile']) for _, user in batch.iterrows()]
        results = await process_batch(user_profiles, llm_caller)
        with open(args.output_path, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
