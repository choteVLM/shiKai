#!/usr/bin/env python3
import json
import argparse
import os
from typing import List, Dict, Any, Tuple
import sys
import re
from abc import ABC, abstractmethod

# Define the models we want to try
OPENAI_PREFERRED_MODELS = ["gpt-3.5-turbo", "gpt-4"]
GEMINI_PREFERRED_MODELS = ["gemini-pro", "gemini-pro-vision"]

class ModelAPI(ABC):
    """Abstract base class for model APIs."""
    
    @abstractmethod
    def setup(self, api_key: str = None):
        """Set up the API with the given key."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        pass
    
    @abstractmethod
    def select_model(self, model_name: str = None):
        """Select a model to use."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response to the prompt."""
        pass

class OpenAIAPI(ModelAPI):
    """OpenAI API implementation."""
    
    def setup(self, api_key: str = None):
        import openai
        self.openai = openai
        
        # Get API key from environment or parameter
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                self.api_key = input("Enter your OpenAI API key: ")
                print("Tip: Set the OPENAI_API_KEY environment variable to avoid entering the key each time.")
        
        self.openai.api_key = self.api_key
        self.model = None
    
    def list_models(self) -> List[Dict[str, Any]]:
        try:
            models_response = self.openai.models.list()
            models = [model.id for model in models_response.data]
            # Filter to only include chat models
            models = [model for model in models if "gpt" in model.lower()]
            return models
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return OPENAI_PREFERRED_MODELS
    
    def select_model(self, model_name: str = None):
        if model_name:
            self.model = model_name
            return
        
        available_models = self.list_models()
        print("\nAvailable OpenAI Models:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = int(input("\nEnter the number of the model you want to use: "))
                if 1 <= choice <= len(available_models):
                    self.model = available_models[choice-1]
                    print(f"\nSelected model: {self.model}")
                    break
                else:
                    print("Invalid choice. Please enter a number from the list above.")
            except ValueError:
                print("Please enter a valid number.")
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing cricket video data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying OpenAI: {e}"

class GeminiAPI(ModelAPI):
    """Google Gemini API implementation."""
    
    def setup(self, api_key: str = None):
        import google.generativeai as genai
        self.genai = genai
        
        # Get API key from environment or parameter
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                self.api_key = input("Enter your Gemini API key: ")
                print("Tip: Set the GEMINI_API_KEY environment variable to avoid entering the key each time.")
        
        self.genai.configure(api_key=self.api_key)
        self.model = None
    
    def list_models(self) -> List[Dict[str, Any]]:
        try:
            available_models = list(self.genai.list_models())
            return available_models
        except Exception as e:
            print(f"Error listing Gemini models: {e}")
            return GEMINI_PREFERRED_MODELS
    
    def select_model(self, model_name: str = None):
        if model_name:
            self.model = self.genai.GenerativeModel(model_name)
            return
        
        available_models = self.list_models()
        print("\nAvailable Gemini Models:")
        for i, m in enumerate(available_models, 1):
            print(f"{i}. {m.name}")
            print(f"   Description: {m.description}")
            print(f"   Supported generation methods: {', '.join(m.supported_generation_methods)}")
            print()
        
        while True:
            try:
                choice = int(input("\nEnter the number of the model you want to use: "))
                if 1 <= choice <= len(available_models):
                    selected_model = available_models[choice-1]
                    print(f"\nSelected model: {selected_model.name}")
                    self.model = self.genai.GenerativeModel(selected_model.name)
                    break
                else:
                    print("Invalid choice. Please enter a number from the list above.")
            except ValueError:
                print("Please enter a valid number.")
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text
            else:
                return "Error: No text in Gemini response"
        except Exception as e:
            return f"Error querying Gemini: {e}"

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the JSON data from the file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_all_available_models() -> Dict[str, List[str]]:
    """Get all available models from all providers."""
    available_models = {}
    
    # Try to get OpenAI models
    openai_api = OpenAIAPI()
    try:
        openai_api.setup()
        available_models["OpenAI"] = openai_api.list_models()
    except Exception as e:
        print(f"Error setting up OpenAI API: {e}")
        available_models["OpenAI"] = OPENAI_PREFERRED_MODELS
    
    # Try to get Gemini models
    gemini_api = GeminiAPI()
    try:
        gemini_api.setup()
        available_models["Gemini"] = gemini_api.list_models()
    except Exception as e:
        print(f"Error setting up Gemini API: {e}")
        available_models["Gemini"] = GEMINI_PREFERRED_MODELS
    
    return available_models

def select_provider_and_model() -> Tuple[str, ModelAPI]:
    """Let the user select a provider and model."""
    providers = {
        "OpenAI": OpenAIAPI,
        "Gemini": GeminiAPI
    }
    
    print("Available AI providers:")
    for i, provider in enumerate(providers.keys(), 1):
        print(f"{i}. {provider}")
    
    choice = input("Select a provider (number): ")
    try:
        provider_index = int(choice) - 1
        provider_name = list(providers.keys())[provider_index]
    except (ValueError, IndexError):
        print("Invalid selection. Using OpenAI as default.")
        provider_name = "OpenAI"
    
    # Create the API instance
    api = providers[provider_name]()
    api.setup()
    
    # Select a model
    api.select_model()
    
    return provider_name, api

def interactive_chat(data: List[Dict[str, Any]], provider_name: str, api: ModelAPI):
    """Run an interactive chat session with the selected model."""
    print("\n" + "="*50)
    print(f"Interactive chat session with {provider_name}")
    print("Type 'exit', 'quit', or 'q' to end the session")
    print("="*50 + "\n")
    
    while True:
        query = input("\nEnter your query about the cricket video: ")
        
        # Check if the user wants to exit
        if query.lower() in ['exit', 'quit', 'q']:
            print("Ending chat session. Goodbye!")
            break
        
        # Prepare the prompt
        prompt = f"""
        You are analyzing a cricket video with the following scene descriptions:
        
        {json.dumps(data, indent=2)}
        
        User query: {query}
        
        Please provide a response that includes relevant timestamps from the video that match the query.
        Format your response as a JSON object with the following structure:
        {{
            "query": "the user's query",
            "relevant_scenes": [
                {{
                    "time_stamp": "timestamp from the video",
                    "description": "brief description of what happens in this scene",
                    "relevance": "why this scene is relevant to the query"
                }},
                ...
            ],
            "summary": "a brief summary of the findings"
        }}
        """
        
        # Query the model
        print(f"\nQuerying {provider_name}...")
        response = api.generate(prompt)
        
        # Try to parse the response as JSON for better formatting
        try:
            response_json = json.loads(response)
            print("\nResponse:")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("\nResponse:")
            print(response)

def main():
    parser = argparse.ArgumentParser(description="Query cricket video data using AI models")
    parser.add_argument("--file", "-f", default="results/RR_batting_10min_world_state_history.json",
                        help="Path to the JSON file containing video data")
    parser.add_argument("--provider", "-p", choices=["OpenAI", "Gemini"],
                        help="AI provider to use (OpenAI, Gemini)")
    parser.add_argument("--model", "-m", help="Model to use")
    parser.add_argument("--query", "-q", help="Query to run against the data")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode after processing the initial query")
    
    args = parser.parse_args()
    
    # Load the data
    data = load_data(args.file)
    
    # If provider or model is not specified, ask the user
    if not args.provider or not args.model:
        provider_name, api = select_provider_and_model()
        args.provider = provider_name
    else:
        # Create the appropriate API instance
        if args.provider.lower() == "openai":
            api = OpenAIAPI()
        elif args.provider.lower() == "gemini":
            api = GeminiAPI()
        else:
            print(f"Unknown provider: {args.provider}")
            sys.exit(1)
        
        # Set up the API
        api.setup()
        
        # Select the model
        api.select_model(args.model)
    
    # If a query is specified, process it
    if args.query:
        # Prepare the prompt
        prompt = f"""
        You are analyzing a cricket video with the following scene descriptions:
        
        {json.dumps(data, indent=2)}
        
        User query: {args.query}
        
        Please provide a response that includes relevant timestamps from the video that match the query.
        Format your response as a JSON object with the following structure:
        {{
            "query": "the user's query",
            "relevant_scenes": [
                {{
                    "time_stamp": "timestamp from the video",
                    "description": "brief description of what happens in this scene",
                    "relevance": "why this scene is relevant to the query"
                }},
                ...
            ],
            "summary": "a brief summary of the findings"
        }}
        """
        
        # Query the model
        print(f"\nQuerying {args.provider}...")
        response = api.generate(prompt)
        
        # Try to parse the response as JSON for better formatting
        try:
            response_json = json.loads(response)
            print("\nResponse:")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("\nResponse:")
            print(response)
    
    # If interactive mode is enabled or no query was specified, start an interactive session
    if args.interactive or not args.query:
        interactive_chat(data, args.provider, api)

if __name__ == "__main__":
    main() 