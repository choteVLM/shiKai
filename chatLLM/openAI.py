import os
from typing import List, Dict, Any

from chatLLM.base_model import chatModel


class openAI(chatModel):
    """openAI API implementation."""
    
    def __init__(self, api_key: str = None):
        import openai
        self.openai = openai
        
        # Get API key from environment or parameter
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_CHAT_API_KEY")
            if not self.api_key:
                self.api_key = input("Enter your OpenAI API key: ")
                print("Tip: Set the OPENAI_API_KEY environment variable to avoid entering the key each time.")
        
        self.openai.api_key = self.api_key
        self.model = None
        self.OPENAI_PREFERRED_MODELS = ["gpt-3.5-turbo", "gpt-4"]

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            models_response = self.openai.models.list()
            models = [model.id for model in models_response.data]
            # Filter to only include chat models
            models = [model for model in models if "gpt" in model.lower()]
            return models
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return self.OPENAI_PREFERRED_MODELS
    
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
                    {"role": "system", "content": "You are a helpful assistant analyzing video data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying OpenAI: {e}"