import feedparser
import requests
import io
import os
from PIL import Image
from datetime import datetime
from huggingface_hub import InferenceClient
from typing import Optional

def load_rss_feed(rss_url: str = "https://buttondown.email/ainews/rss") -> str:
    """Load and parse RSS feed, returning just the description text of the latest item"""
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return "No news available"
    
    # Get first entry's description
    desc = feed.entries[0].description
    
    # Extract Reddit Recap section if present
    start_idx = desc.find("<h1 id=\"ai-reddit-recap\">")
    end_idx = desc.find("<h1 id=\"ai-discord-recap\">")
    if start_idx != -1 and end_idx != -1:
        desc = desc[start_idx:end_idx]
    
    return desc

def get_gemma_response(prompt: str, api_key: str) -> str:
    """Get response from Gemma model"""
    client = InferenceClient(api_key=api_key)
    
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    stream = client.chat.completions.create(
        model="google/gemma-1.1-7b-it",
        messages=messages,
        max_tokens=500,
        stream=True
    )
    
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content
    return response

def generate_image(prompt: str, api_key: str, num_of_tries: int) -> str:
    """Generate image from prompt and return filename"""
    if num_of_tries <= 0:
        return "20250109_102000.png"

    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code == 200 and response.headers['Content-Type'].startswith('image/'):
        try:
            image = Image.open(io.BytesIO(response.content))
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
            image.save(f"pictures/{filename}")
            return filename
        except Exception as e:
            print("Failed to generate image. With error: ", e)
            return generate_image(prompt, api_key, num_of_tries-1)
    else:
        print("Failed to generate image. With status code: ", response.status_code)
        return generate_image(prompt, api_key, num_of_tries-1)

def update_readme(summary: str, image_filename: str):
    """Update README.md with new content"""
    readme_template = f"""
# Today's AI News

![Todays Image](pictures/{image_filename})

{summary}
"""
    with open('README.md', 'w') as f:
        f.write(readme_template)

def main(huggingface_api_key: str):
    # Load RSS feed content
    news_content = load_rss_feed()
    
    # Get news summary from Gemma
    summary_prompt = f"""
    From the following list of news items, pick the most interesting few and do a very short but appealing summary of it.
    {news_content}
    """
    summary = get_gemma_response(summary_prompt, huggingface_api_key)
    
    # Get image generation prompt from Gemma
    image_prompt_gen = f"""
    Of the following which is the most shocking catchy clickbait title, 
    pick only one, state it, then proceed with a visual description (no infographic, no explicit text in the image)
    {summary}
    """
    image_prompt = get_gemma_response(image_prompt_gen, huggingface_api_key)
    print(image_prompt)
    
    # Generate image
    image_filename = generate_image(image_prompt, huggingface_api_key, 3)
    
    # Update README
    update_readme(summary, image_filename)

if __name__ == "__main__":
    # from dotenv import load_dotenv # only for local testing    
    # load_dotenv()

    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not huggingface_api_key:
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        
    main(huggingface_api_key)

