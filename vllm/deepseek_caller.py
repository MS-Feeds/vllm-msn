from typing import Dict, List, Optional, Union
import openai
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class QwenCaller:
    """A class to handle API calls to Qwen models through OpenAI API."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize the Qwen caller.
        
        Args:
            api_key (str): The API key for authentication
            base_url (Optional[str]): The base URL for the API. If None, uses default OpenAI URL
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def generate(
        self,
        prompt: str,
        model: str = "qwen-3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
    ) -> Union[str, Dict]:
        """Generate text using the Qwen model.
        
        Args:
            prompt (str): The input prompt
            model (str): The model to use (default: qwen-3)
            temperature (float): Controls randomness in the output
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Controls diversity via nucleus sampling
            frequency_penalty (float): Penalizes repeated tokens
            presence_penalty (float): Penalizes new tokens
            stop (Optional[Union[str, List[str]]]): Stop sequences
            stream (bool): Whether to stream the response
            
        Returns:
            Union[str, Dict]: The generated text or streaming response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen-3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
    ) -> Union[str, Dict]:
        """Generate text using the Qwen model with a list of messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
            model (str): The model to use (default: qwen-3)
            temperature (float): Controls randomness in the output
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Controls diversity via nucleus sampling
            frequency_penalty (float): Penalizes repeated tokens
            presence_penalty (float): Penalizes new tokens
            stop (Optional[Union[str, List[str]]]): Stop sequences
            stream (bool): Whether to stream the response
            
        Returns:
            Union[str, Dict]: The generated text or streaming response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise 