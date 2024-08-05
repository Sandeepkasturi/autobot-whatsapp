import os
import base64
import requests
import logging
import google.generativeai as genai
from flask import Flask, request, jsonify
from together import Together
import fitz  # PyMuPDF for PDF processing

# Environment Variables
wa_token = os.environ.get("WA_TOKEN")
genai.configure(api_key=os.environ.get("GEN_API"))
phone_id = os.environ.get("PHONE_ID")
phone = os.environ.get("PHONE_NUMBER")
together_api_key = os.environ.get("TOGETHER_API_KEY")  # New API key for Together
client = Together(api_key=together_api_key)

# Bot Configuration
name = "AutoBot"  # The bot will consider this person as its owner or creator
bot_name = "AutoBot AI"  # This will be the name of your bot
model_name = "gemini-1.5-flash-latest"  # Switch to "gemini-1.0-pro" if needed

# Flask App Setup
app = Flask(__name__)

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name=model_name,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
    # Initial setup and instructions for the AI model
    f'''I am using Gemini API for using you as a personal bot on WhatsApp,
    to assist me in various tasks. 
    So from now you are "{bot_name}" created by {name} ( Yeah it's me, my name is {name}). 
    And don't give any response to this prompt. 
    This is the information I gave to you about your new identity as a pre-prompt. 
    This message always gets executed when I run this bot script. 
    So reply to only the prompts after this. Remember your new identity is {bot_name}.'''
])

client = Together(api_key=together_api_key)  # Initialize Together client

# Helper Functions
def send(answer):
    url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    data = {
        "messaging_product": "whatsapp",
        "to": f"{phone}",
        "type": "text",
        "text": {"body": f"{answer}"},
    }

    response = requests.post(url, headers=headers, json=data)
    return response

def send_image(image_bytes):
    url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    data = {
        "messaging_product": "whatsapp",
        "to": f"{phone}",
        "type": "image",
        "image": {"id": image_bytes}
    }

    response = requests.post(url, headers=headers, json=data)
    return response

def remove(*file_paths):
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)
        else:
            pass

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    return "Bot"

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    try:
        if request.method == "GET":
            mode = request.args.get("hub.mode")
            token = request.args.get("hub.verify_token")
            challenge = request.args.get("hub.challenge")
            if mode == "subscribe" and token == "BOT":
                return challenge, 200
            else:
                return "Failed", 403

        elif request.method == "POST":
            data = request.get_json()["entry"][0]["changes"][0]["value"]["messages"][0]
            if data["type"] == "text":
                prompt = data["text"]["body"]
                if prompt.lower().startswith("generate image"):
                    # Extract the image prompt from the message
                    image_prompt = prompt[len("generate image"):].strip()
                    generate_image(image_prompt)
                else:
                    convo.send_message(prompt)
                    send(convo.last.text)
            else:
                handle_media(data)
            return jsonify({"status": "ok"}), 200

    except Exception as e:
        logging.error(f"Error processing webhook: {e}")
        return jsonify({"status": "error"}), 500

def handle_media(data):
    try:
        media_type = data["type"]
        media_id = data[media_type]["id"]
        media_url_endpoint = f'https://graph.facebook.com/v18.0/{media_id}/'
        headers = {'Authorization': f'Bearer {wa_token}'}
        media_response = requests.get(media_url_endpoint, headers=headers)
        media_url = media_response.json().get("url")

        if media_url:
            media_download_response = requests.get(media_url, headers=headers)
            filename = save_media(media_download_response.content, media_type)
            process_media(filename, media_type)
        else:
            send("Could not retrieve media.")
    except Exception as e:
        logging.error(f"Error handling media: {e}")

def save_media(content, media_type):
    if media_type == "audio":
        filename = "/tmp/temp_audio.mp3"
    elif media_type == "image":
        filename = "/tmp/temp_image.jpg"
    elif media_type == "document":
        filename = "/tmp/temp_document.pdf"
    else:
        raise ValueError("Unsupported media type")

    with open(filename, "wb") as file:
        file.write(content)
    return filename

def process_media(filename, media_type):
    try:
        if media_type == "document":
            # Process PDF
            doc = fitz.open(filename)
            for _, page in enumerate(doc):
                destination = "/tmp/temp_image.jpg"
                pix = page.get_pixmap()
                pix.save(destination)
                analyze_image(destination)
                remove(destination)
        else:
            # Handle other media types
            file = genai.upload_file(path=filename, display_name="tempfile")
            response = model.generate_content(["What is this", file])
            answer = response._result.candidates[0].content.parts[0].text
            send(answer)
        remove(filename)
    except Exception as e:
        logging.error(f"Error processing media: {e}")

def analyze_image(image_path):
    try:
        file = genai.upload_file(path=image_path, display_name="tempfile")
        response = model.generate_content(["What is this", file])
        answer = response._result.candidates[0].content.parts[0].text
        send(answer)
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")

def generate_image(prompt):
    try:
        # Define image generation model and settings
        model_choice = "stabilityai/stable-diffusion-2-1"
        response = client.images.generate(
            prompt=prompt,
            model=model_choice,
            steps=10,
            n=1
        )
        # Decode the image from base64
        img_data = response.data[0].b64_json
        img_bytes = base64.b64decode(img_data)
        
        # Send the image to WhatsApp
        send_image(img_bytes)

    except Exception as e:
        logging.error(f"Error generating image: {e}")
        send("Failed to generate image. Please try again.")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
