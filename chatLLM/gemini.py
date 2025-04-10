import os
from typing import List, Dict, Any

import google.generativeai as genai

from chatLLM.base_model import chatModel



class gemini(chatModel):
    """Google Gemini API implementation."""
    
    def __init__(self, api_key: str = None):

        self.genai = genai
        
        # Get API key from environment or parameter
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GEMINI_CHAT_API_KEY")
            if not self.api_key:
                self.api_key = input("Enter your Gemini API key: ")
                print("Tip: Set the GEMINI_API_KEY environment variable to avoid entering the key each time.")
        
        self.genai.configure(api_key=self.api_key)
        self.model = None
        self.GEMINI_PREFERRED_MODELS = ["gemini-pro", "gemini-pro-vision"]
    
    def list_models(self) -> List[Dict[str, Any]]:
        try:
            available_models = list(self.genai.list_models())
            return available_models
        except Exception as e:
            print(f"Error listing Gemini models: {e}")
            return self.GEMINI_PREFERRED_MODELS
    
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