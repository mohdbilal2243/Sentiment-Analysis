import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env into the environment
load_dotenv()

# Access specific variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Creating Connection Object (Client) to talk to Groq's AI
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Folder creation
def create_folder(path):
    os.makedirs(path, exist_ok=True)

# Function For Sentiment Analysis
def analyze_sentiment_groq(text, filename, base_output_dir = "result"):
    """
    Perform Sentiment analysis using Groq LLM
    output : Positive / Neutral / Negative
    """

    try:
        # Check if already Processed
        output_dir = os.path.join(base_output_dir, filename, "sentiment")
        create_folder(output_dir)

        file_path = os.path.join(output_dir, f"{filename}_sentiment.txt")

        if not os.path.exists(file_path):

            print("Running Sentiment Analysis (Groq)...")

            prompt = f"""
            You are a Strict Sentiment Classifier.
    
            Rules : 
            - output ONLY one word
            - Choose from : Positive, Neutral, Negative
            - Do NOT Explain
    
            Text: 
            {text}
            """

            response = client.chat.completions.create(
                model = MODEL_NAME,
                messages = [
                {"role": "system", "content": "You are a sentiment classifier."},
                {"role": "user", "content": prompt}
                ],
                temperature = 0   # Deterministic Output Means For same input it does not change the Output
            )

            sentiment = response.choices[0].message.content.strip()

            # Save result
            with open(file_path, "w", encoding = "utf-8") as f:
                f.write(sentiment)

            print(f"Sentiment : {sentiment}")
            print()
            print(f"Saved at : {file_path}")

        else:
            print("Sentiment already esists, Skipping API call...")

            # read existing result
            with open(file_path, "r", encoding = "utf-8") as f:
                sentiment = f.read().strip()

        return sentiment

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return None


