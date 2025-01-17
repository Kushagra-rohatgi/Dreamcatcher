from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Initialize Hugging Face pipelines
# Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Text Generation Pipeline
text_generator = pipeline("text-generation", model="gpt2", max_length=100)

# Stable Diffusion Pipeline for Image Generation
image_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
image_pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def analyze_sentiment_with_huggingface(text):
    """
    Perform sentiment analysis on the user-provided text.
    """
    result = sentiment_analyzer(text)[0]
    sentiment = result['label']
    score = result['score']
    return sentiment, score


def generate_image_from_text(prompt):
    """
    Generate an image using a Stable Diffusion model based on the user's input.
    Always includes the keyword 'animated' in the backend.
    """
    print("Generating your dream image... This may take a moment.")
    backend_prompt = f"{prompt}, animated"  # Add "animated" to the prompt
    image = image_pipe(backend_prompt).images[0]
    image_path = "dream_image.png"
    image.save(image_path)
    print(f"Image saved as {image_path}")
    return image_path


def generate_dynamic_remark(keywords, sentiment, feeling, additional_comments):
    """
    Generate a dynamic remark based on the sentiment, keywords, and user input.
    """
    input_for_generation = (
        f"Dream Keywords: {keywords}\n"
        f"Sentiment: {sentiment}\n"
        f"User Feeling: {feeling}\n"
        f"Additional Comments: {additional_comments}\n\n"
        "Based on this input, craft a unique and meaningful interpretation of the user's dream:"
    )

    response = text_generator(input_for_generation)[0]['generated_text']
    # Trim the response to avoid excessive length
    remark = response.split("Based on this input")[-1].strip()
    return remark


def interactive_dream_analysis():
    """
    Interactive session to analyze user input, perform sentiment analysis,
    and generate a personalized dream image with unique feedback.
    """
    print("Welcome to Dreamcatcher!")
    
    # Step 1: Gather dream keywords
    keywords = []
    print("Enter keywords describing your dream (press Enter after each keyword, type 'done' when finished):")
    while True:
        keyword = input("Keyword: ").strip()
        if keyword.lower() == "done":
            break
        if keyword:
            keywords.append(keyword)
    
    # Combine keywords into a prompt
    dream_prompt = ", ".join(keywords)

    # Step 2: Ask user for additional context
    user_feeling = input("How did you feel during this dream? ").strip()
    additional_comments = input("Do you have any other thoughts or details to share? ").strip()
    
    # Full user input for sentiment analysis
    full_input = f"Dream Keywords: {dream_prompt}\nFeeling: {user_feeling}\nAdditional Comments: {additional_comments}"
    
    # Step 3: Perform sentiment analysis
    sentiment, score = analyze_sentiment_with_huggingface(full_input)
    print(f"Analyzed Sentiment: {sentiment} (Confidence: {score:.2f})")

    # Step 4: Generate a unique remark using generative AI
    remark = generate_dynamic_remark(dream_prompt, sentiment, user_feeling, additional_comments)
    print(f"Remark: {remark}")
    
    # Step 5: Generate dream image
    generate_image_from_text(dream_prompt)


# Run the interactive session
interactive_dream_analysis()
