import requests
import time
import os

class DalleService:
    def __init__(self):
        self.url = "https://api.openai.com/v1/images/generations"
        api_key = os.getenv("Your OpenAI_API_Key")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _gpt_response(self, prompt):
        data = {
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "model": "dall-e-3"
        }
        try:
            response = requests.post(self.url, headers=self.headers, json=data)
            if response.status_code == 200:
                image_url = response.json()['data'][0]['url']
                print(f"Generated Image URL: {image_url}")
                return image_url
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def gpt_with_retry(self, prompt, max_retries=8, sleep_seconds=3):
        for attempt in range(max_retries):
            result = self._gpt_response(prompt)
            if result is not None:
                return result
            print(f"Retrying... ({attempt + 1}/{max_retries})")
            time.sleep(sleep_seconds)
        print("All retries failed.")
        return None

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image successfully downloaded: {save_path}")
        else:
            print(f"Error: Unable to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_dalle3(prompt: str, output_path: str):
    dalle3 = DalleService()
    image_url = dalle3.gpt_with_retry(prompt)
    download_image(image_url, output_path)

