#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Analyzer Module for Content Recommender

This module handles the analysis of user browsing/search history 
to infer user interests using DeepSeek LLM.

Author: Zoey
Date: May 12, 2025
"""

import json
import logging
import requests
import time
import random
import datetime
from typing import List, Dict, Any, Callable
from openai import OpenAI
# from content_retriever import search_bing_web
import pandas as pd
import tqdm
# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def prepare_history_analysis_prompt(user_profile: Dict) -> Dict:
    """
    Prepare a prompt for the DeepSeek LLM to analyze user profile and infer interests.
    
    Args:
        user_profile: Dictionary containing user profile data including interests
        
    Returns:
        Dict: Formatted prompt for the DeepSeek API
    """
    # Get current date information for timestamp context
    current_date = datetime.datetime.now()
    current_year = current_date.strftime("%Y")
    current_month = current_date.strftime("%B")  # Full month name    # Extract user profile information
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
    
    interests_list = "\n".join(interests_text) if interests_text else "No explicit interests provided"

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
                     "content": """You are an advanced AI model specializing in analyzing user profiles to infer high-confidence, diverse, and valuable niche interests for ongoing tracking and content recommendation.\n\nYour recommendations must:\n- Be based on clear evidence from the user's profile\n- Have ongoing relevance, with new content regularly published\n- Be uniquely tailored to the user's actual interests (not generic topics)\n- Enable highly personalized content recommendations\n\nWhen analyzing user profile:\n1. Prioritize recent and repeated activities\n2. Identify patterns indicating sustained or growing interest\n3. Focus on specific niches, not broad categories\n4. Recommend topics where tracking future developments will benefit the user\n5. Ensure each recommended interest is meaningfully different from the others\n\nDo NOT recommend:\n- Adult, offensive, or crime-related topics\n- One-off or transient interests\n- Topics unlikely to have fresh future content\n- Generic, non-personalized categories\n- Low-confidence inferences based on limited evidence\n"""
       },
            {
                "role": "user",
                "content": prompt_content
            }
        ]
    }

# def resolve_ambiguities(inferred_interests: List[Dict], bing_api_key: str) -> List[Dict]:
#     """
#     Resolve any ambiguities in inferred interests by using Bing Search.
    
#     Args:
#         inferred_interests: List of inferred interests with potential ambiguities
#         bing_api_key: API key for Bing search
        
#     Returns:
#         List[Dict]: Enhanced interests with ambiguities resolved
#     """
#     enriched_interests = []
    
#     for interest in inferred_interests:
#         # Check if the interest has ambiguities that need clarification
#         if interest.get('ambiguous', False) and 'clarification_needed' in interest:
#             ambiguous_topic = interest['clarification_needed']
#             logger.info(f"Resolving ambiguity: {ambiguous_topic}")
            
#             # Call Bing API to get context
#             context = search_bing_web(ambiguous_topic, bing_api_key)
            
#             # Enrich the interest with the context
#             interest['context'] = context
#             interest['ambiguous'] = False
            
#         enriched_interests.append(interest)
    
#     return enriched_interests

class Qwen3Caller:
    def __init__(self, base_url: str = 'http://127.0.0.1:8000/v1'):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def __call__(self, messages: Dict) -> Dict:
        """
        Make the class callable to handle API requests.
        
        Args:
            messages: Dictionary containing the messages for the API request
            
        Returns:
            Dict: API response containing the model's output
        """
        try:
            # Extract the messages list from the dictionary
            messages_list = messages.get("messages", [])
            completion = self.client.chat.completions.create(
                # model="/nvmedata/hf_checkpoints/Qwen3-8B/",
                model="/nvmedata/hf_checkpoints/Qwen3-32B/",
                messages=messages_list,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            return completion.model_dump()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

def analyze_user_history(user_profile: Dict, qwen3_caller: Callable) -> List[Dict]:
    """
    Analyze user profile to infer interests using DeepSeek LLM.
    
    Args:
        user_profile: Dictionary containing user profile data
        qwen3_caller: Function to call DeepSeek API
        
    Returns:
        List[Dict]: Inferred user interests
    """
    # Prepare prompt for profile analysis
    prompt = prepare_history_analysis_prompt(user_profile)
    
    # Call Qwen3 LLM for analysis
    # print(prompt)
    response = qwen3_caller(prompt)
    try:
        # print(response.keys())
        # breakpoint()
        assert 'choices' in response, "Invalid response format: missing 'choices'"
        # Extract interest recommendations from response
        if "choices" not in response or not response["choices"]:
            logger.error("Invalid response format: missing 'choices'")
            return []

        content = response["choices"][0]["message"]["content"]
        # print(content)
        # skip the <think> </think> part
        # content = content.split("<think>")[1].split("</think>")[1].strip()  
        # print(content)
        # breakpoint()
        logger.info(f"Raw LLM response: {content[:200]}...")  # Log the first part of the response
        
        # Check if the response is "can't infer"
        if "can't infer" in content.lower():
            logger.warning("LLM couldn't infer any interests from the provided history")
            return []
        
        # Extract JSON from the content
        json_str = content
        # First, try to find JSON block enclosed in triple backticks
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
            logger.info("Extracted JSON from ```json``` block")
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
            logger.info("Extracted JSON from ``` block")
        
        # If still not found, search for text within square brackets (for array)
        if not json_str or (not json_str.startswith("[") and not json_str.endswith("]")):
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                logger.info(f"Extracted JSON using bracket detection: {start_idx}-{end_idx}")
        
        logger.info(f"JSON string to parse: {json_str[:200]}...")
        
        try:
            inferred_interests = json.loads(json_str)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON decode error: {json_err}. Attempting to clean and retry...")
            # Try to clean up common JSON formatting issues
            cleaned_json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
            cleaned_json_str = cleaned_json_str.replace("\\", "\\\\")  # Escape backslashes
            inferred_interests = json.loads(cleaned_json_str)
            logger.info("Successfully parsed JSON after cleanup")
        
        # Validate format
        if not isinstance(inferred_interests, list):
            logger.error(f"Invalid interests format: not a list. Got type: {type(inferred_interests)}")
            if isinstance(inferred_interests, dict):
                # Try to handle case where the response is a single object instead of a list
                logger.warning("Received a single object instead of a list. Converting to a list.")
                inferred_interests = [inferred_interests]
            else:
                return []
            
        # Ensure all required fields are present
        validated_interests = []
        for interest in inferred_interests:
            if "name" in interest and "reason" in interest:
                # Handle missing queries field
                if "queries" not in interest or not interest["queries"]:
                    interest["queries"] = [f"latest {interest['name']} news"]
                    logger.warning(f"Generated default query for interest: {interest['name']}")
                
                # Ensure queries is a list
                if not isinstance(interest["queries"], list):
                    interest["queries"] = [interest["queries"]]
                    logger.warning(f"Converted queries to list for interest: {interest['name']}")
                
                # Ensure description exists
                if "description" not in interest or not interest["description"]:
                    interest["description"] = interest["reason"]
                    logger.warning(f"Used reason as description for interest: {interest['name']}")
                    
                # Ensure keywords exist
                if "keywords" not in interest or not interest["keywords"]:
                    interest["keywords"] = interest["name"].split()
                    logger.warning(f"Generated keywords from name for interest: {interest['name']}")
                
                # Ensure strength is a number between 1 and 10
                if "strength" not in interest or not isinstance(interest["strength"], (int, float)):
                    interest["strength"] = 5
                    logger.warning(f"Used default strength (5) for interest: {interest['name']}")
                else:
                    interest["strength"] = min(max(interest["strength"], 1), 10)
                    
                validated_interests.append(interest)
            else:
                logger.warning(f"Skipped interest due to missing required fields: {interest}")
        
        logger.info(f"Successfully validated {len(validated_interests)} interests")
        return validated_interests
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing inferred interests: {e}")
        breakpoint()
        logger.error(f"Response content: {response.get('choices', [{}])[0].get('message', {}).get('content', '')[:500]}")
        return []

def parse_raw_user_data(raw_data: str):
    user_profile = {}

    # Step 1: Split top-level fields
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
                keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                interests.append({
                    "category": category.strip(),
                    "keywords": keywords
                })
            user_profile["Interests"] = interests

    return user_profile


if __name__ == "__main__":
    users_path = '/nvmedata/chenw/genz/genz_users_20k_format.tsv'
    df = pd.read_csv(users_path, sep='\t')
    qwen_caller = Qwen3Caller(base_url="http://127.0.0.1:8000/v1")
    # Start the performance test
    start_time = time.time()
    for i in tqdm.tqdm(range(len(df))):
        user = df.iloc[i]
        # convert to dict
        user = user.to_dict()
        user_profile = parse_raw_user_data(user['profile'])
        analyze_user_history(user_profile, qwen_caller)
        # break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


    