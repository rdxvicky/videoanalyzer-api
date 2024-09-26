from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import tempfile
import requests
from moviepy.editor import VideoFileClip

# Initialize FastAPI
app = FastAPI()

# Initialize Vertex AI with your project and location
PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load the Gemini Pro Vision model with caching
def load_model():
    try:
        model = GenerativeModel("gemini-1.5-flash-002")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Cache the loaded model to avoid reloading on every request
multimodal_model = load_model()

# Pydantic models for request and response
class VideoAnalysisRequest(BaseModel):
    cdn_url: str

class VideoAnalysisResponse(BaseModel):
    result: dict  # Structured JSON output

# Function to download video, store temporarily, and extract metadata using moviepy
def download_and_extract_metadata(cdn_url):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            video_response = requests.get(cdn_url, stream=True)
            if video_response.status_code == 200:
                for chunk in video_response.iter_content(chunk_size=8192):
                    tmp_video.write(chunk)
            else:
                raise ValueError(f"Failed to download video from CDN URL. Status code: {video_response.status_code}")
            video_path = tmp_video.name

        with VideoFileClip(video_path) as video:
            video_duration_sec = video.duration
            video_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB

        return video_size, video_duration_sec, video_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or processing video: {e}")

# Function to delete the temporary file after use
def delete_temp_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting temporary file: {e}")

# Function to convert video content to a Part object
def convert_video_to_part(video_path):
    try:
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
        video_part = Part.from_data(mime_type="video/mp4", data=video_data)
        return video_part
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting video to Part: {e}")

# Helper function to generate content using the model
def generate_content(model: GenerativeModel, prompt: str, video_part: Part) -> dict:
    try:
        response = model.generate_content(
            [prompt, video_part],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 2048
            },
            stream=True
        )
        final_response = [resp.text for resp in response if resp.text]
        if not final_response:
            raise ValueError("Model returned an empty response.")
        return {"data": final_response}
    except Exception as e:
        raise RuntimeError(f"Content generation failed: {e}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to Paigeon Video Analyzer"}

# Improved prompts and structured JSON output for different endpoints

# 1. Tags Endpoint
@app.post("/analyze-video/tags", response_model=VideoAnalysisResponse)
async def generate_tags(request: VideoAnalysisRequest):
    """
    Generate relevant and popular tags for the video content with probabilities.
    """
    prompt = (
        "Based on the video content, generate a list of 10 relevant tags that accurately describe the main themes, objects, actions, and concepts present. "
        "For each tag, provide a probability score between 0 and 1 indicating its relevance and prominence in the video. "
        "Structure the output as a JSON array of objects, each containing a 'tag' and a 'probability' field. "
        "Ensure tags are specific, varied, and cover different aspects of the video content."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 2. Emotion Detection Endpoint
@app.post("/analyze-video/emotions", response_model=VideoAnalysisResponse)
async def detect_emotions(request: VideoAnalysisRequest):
    """
    Identify and describe the emotions displayed by individuals in the video, with time frames.
    """
    prompt = (
        "Analyze the provided video data to identify emotions displayed by individuals. "
        "Focus on facial expressions, body language, and context to infer emotional states. "
        "For each identified emotion, provide:\n"
        "1. A description of the emotion (e.g., joy, anger, surprise)\n"
        "2. Any relevant context or cause for the emotion\n"
        "3. An approximate time frame (in seconds) when the emotion is observed\n"
        "Structure the response as a list of emotional observations, each with these three elements. "
        "If no clear emotions are detectable, state this and provide possible reasons why."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 3. Description Endpoint
@app.post("/analyze-video/description", response_model=VideoAnalysisResponse)
async def analyze_description(request: VideoAnalysisRequest):
    """
    Generate a detailed description of the video content.
    """
    prompt = (
        "Analyze the provided video data and generate a comprehensive description of its content. "
        "Focus on:\n"
        "1. The main subject or theme of the video\n"
        "2. Key events or actions that occur, with approximate time frames (in seconds)\n"
        "3. Notable visual elements, such as setting, characters, or objects\n"
        "4. Any significant audio elements, including dialogue, music, or sound effects\n"
        "5. The overall mood or tone of the video\n"
        "Provide a coherent narrative that captures these elements, giving the reader a clear understanding of the video's content without having viewed it directly."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 4. Scene Segmentation Endpoint
@app.post("/analyze-video/scene-segmentation", response_model=VideoAnalysisResponse)
async def segment_scenes(request: VideoAnalysisRequest):
    """
    Break down the video into distinct scenes and provide a description for each, with time frames.
    """
    prompt = (
        "Analyze the provided video data to segment it into distinct scenes. For each scene:\n"
        "1. Provide a brief description of the scene's content and action\n"
        "2. Note any significant changes in setting, characters, or tone\n"
        "3. Include approximate start and end times (in seconds)\n"
        "4. Highlight any key moments or turning points within the scene\n"
        "Present the scenes in chronological order, ensuring that the segmentation captures the video's narrative or structural flow. "
        "If there are no clear scene changes, describe the video's content in meaningful segments based on action or subject matter."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 5. Transcription Endpoint
@app.post("/analyze-video/transcription", response_model=VideoAnalysisResponse)
async def transcribe_video(request: VideoAnalysisRequest):
    """
    Provide a complete and accurate transcription of all spoken words in the video.
    """
    prompt = (
        "Analyze the provided video and transcribe all spoken words, capturing dialogues, speeches, and verbal interactions. "
        "Include time stamps (in seconds) for when each speech segment begins."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 6. Highlights Endpoint
@app.post("/analyze-video/highlights", response_model=VideoAnalysisResponse)
async def summarize_highlights(request: VideoAnalysisRequest):
    """
    Summarize the key highlights and important moments of the video.
    """
    prompt = (
        "Based on the video content, identify and summarize the key highlights and important moments. "
        "For each highlight, provide a brief description and a time frame (in seconds) for when it occurs."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 7. Shopping Items Endpoint
@app.post("/analyze-video/shopping", response_model=VideoAnalysisResponse)
async def identify_shopping_items(request: VideoAnalysisRequest):
    """
    Identify objects in the video that could be linked to online shopping.
    """
    prompt = (
        "Analyze the provided video content to identify any products, clothing, accessories, or other objects that could be of interest for online shopping. "
        "For each object, provide a brief description and the time frame (in seconds) when it appears."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 8. Action Recognition Endpoint with Time Frames
@app.post("/analyze-video/actions", response_model=VideoAnalysisResponse)
async def recognize_actions(request: VideoAnalysisRequest):
    """
    Describe the main actions or activities occurring in the video with time frames.
    """
    prompt = (
        "Analyze the video data and describe the main actions and activities occurring in the video. Include start and end timestamps (in seconds) for each action."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 9. Sentiment Analysis Endpoint
@app.post("/analyze-video/sentiment", response_model=VideoAnalysisResponse)
async def analyze_sentiment(request: VideoAnalysisRequest):
    """
    Analyze the overall sentiment and emotional tone conveyed in the video.
    """
    prompt = (
        "Analyze the overall sentiment and emotional tone of the video, such as positive, negative, or neutral. "
        "Provide examples from the video to support your analysis."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 10. Language Detection Endpoint
@app.post("/analyze-video/language-detection", response_model=VideoAnalysisResponse)
async def detect_language(request: VideoAnalysisRequest):
    """
    Identify all languages spoken in the video.
    """
    prompt = (
        "Identify all the languages spoken in the video, including any accents or dialects. "
        "Provide the time frames where different languages are used."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 11. Keyword Extraction Endpoint
@app.post("/analyze-video/keywords", response_model=VideoAnalysisResponse)
async def extract_keywords(request: VideoAnalysisRequest):
    """
    Extract key phrases and keywords that summarize the video's content, with importance scores.
    """
    prompt = (
        "Extract key phrases and keywords that summarize the video's content. "
        "For each keyword, provide an importance score between 0 and 1, and explain its relevance briefly."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 12. Optical Character Recognition (OCR) Endpoint
@app.post("/analyze-video/ocr", response_model=VideoAnalysisResponse)
async def extract_text(request: VideoAnalysisRequest):
    """
    Extract all text displayed within the video frames.
    """
    prompt = (
        "Extract all visible text displayed within the video frames, such as signs, labels, subtitles, and any other on-screen text. "
        "Provide the text content along with the time frames (in seconds) when they appear."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 13. Brand Logo Detection Endpoint
@app.post("/analyze-video/brand-logos", response_model=VideoAnalysisResponse)
async def detect_brand_logos(request: VideoAnalysisRequest):
    """
    Identify any brand logos or trademarks appearing in the video.
    """
    prompt = (
        "Identify any brand logos, trademarks, or recognizable brand elements appearing in the video. "
        "For each, provide the time frames (in seconds) when they appear."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 14. Summarization for Social Media Endpoint
@app.post("/analyze-video/social-media-summary", response_model=VideoAnalysisResponse)
async def social_media_summary(request: VideoAnalysisRequest):
    """
    Create a short and engaging summary suitable for sharing on social media platforms.
    """
    prompt = (
        "Create a short, engaging, and catchy summary of the video suitable for sharing on social media platforms. "
        "The summary should capture the essence and highlights of the video in a way that attracts viewers."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 15. Visual Style Analysis Endpoint
@app.post("/analyze-video/visual-style", response_model=VideoAnalysisResponse)
async def analyze_visual_style(request: VideoAnalysisRequest):
    """
    Analyze the visual style and aesthetic elements used in the video.
    """
    prompt = (
        "Analyze the visual style of the video, discussing elements such as color schemes, cinematography, visual effects, and overall aesthetic appeal. "
        "Explain how these elements contribute to the video's mood and message."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 16. Anomaly Detection Endpoint
@app.post("/analyze-video/anomalies", response_model=VideoAnalysisResponse)
async def detect_anomalies(request: VideoAnalysisRequest):
    """
    Detect any unusual or unexpected events in the video.
    """
    prompt = (
        "Analyze the video to detect any unusual or unexpected events or anomalies. "
        "Provide details and time frames (in seconds) for each detected anomaly."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 17. Gesture Recognition Endpoint
@app.post("/analyze-video/gestures", response_model=VideoAnalysisResponse)
async def recognize_gestures(request: VideoAnalysisRequest):
    """
    Identify and interpret significant gestures made by individuals in the video.
    """
    prompt = (
        "Identify and interpret any significant gestures made by individuals in the video. "
        "Provide descriptions and the time frames (in seconds) when they occur."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 18. Topic/Genre Identification Endpoint
@app.post("/analyze-video/topics", response_model=VideoAnalysisResponse)
async def identify_topics(request: VideoAnalysisRequest):
    """
    Determine the main topics and subtopics discussed in the video.
    """
    prompt = (
        "Analyze the video and determine its main topics or genres. "
        "For each topic or genre, provide a brief description of why it is relevant, and list subtopics if applicable. "
        "Structure the output as a JSON array of topics, each containing a 'topic' field and a list of 'subtopics' if available. "
        "Ensure that the topics cover the major themes or subject areas depicted in the video."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)

# 19. Video Moderation Endpoint
@app.post("/analyze-video/moderation", response_model=VideoAnalysisResponse)
async def moderate_content(request: VideoAnalysisRequest):
    """
    Detect and report any inappropriate or restricted content in the video, with time frames and details.
    """
    prompt = (
        "Analyze the video for any inappropriate, offensive, or restricted content, such as violence, nudity, explicit language, or hate speech. "
        "For each instance of such content, provide details including the time frames (in seconds) where it occurs. "
        "Structure the output as a JSON array of objects, each containing a 'type' of inappropriate content and the 'time_frame' when it appears."
    )
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)
