from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, GenerationConfig
import tempfile
import requests
from moviepy.editor import VideoFileClip
import json

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

# Function to clean messy model output and ensure valid JSON
def clean_response(messy_response: list) -> dict:
    # Join response fragments and parse as JSON
    try:
        combined_response = "".join(messy_response)
        cleaned_response = json.loads(combined_response)
        return cleaned_response
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse model response as valid JSON.")

# Helper function to generate structured content using the model
def generate_content(model: GenerativeModel, prompt: str, video_part: Part, schema: dict) -> dict:
    try:
        response = model.generate_content(
            [prompt, video_part],
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,
                max_output_tokens=2048
            ),
            stream=True
        )
        # Clean and structure response data
        messy_response = [resp.text for resp in response if resp.text]
        if not messy_response:
            raise ValueError("Model returned an empty response.")
        return clean_response(messy_response)
    except Exception as e:
        raise RuntimeError(f"Content generation failed: {e}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to Paigeon Video Analyzer"}

# Define schemas for each endpoint

# 1. Tags Endpoint Schema
tags_schema = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "probability": {"type": "number"}
                },
                "required": ["tag", "probability"]
            }
        }
    }
}

# 2. Emotion Detection Schema
emotions_schema = {
    "type": "object",
    "properties": {
        "emotions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "emotion": {"type": "string"},
                    "time_frame": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["emotion", "time_frame"]
            }
        }
    }
}

# 3. Description Schema
description_schema = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
    }
}

# 4. Scene Segmentation Schema
scene_segmentation_schema = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"}
                },
                "required": ["description", "start_time", "end_time"]
            }
        }
    }
}

# 5. Transcription Schema
transcription_schema = {
    "type": "object",
    "properties": {
        "transcriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["text", "time_frame"]
            }
        }
    }
}

# 6. Highlights Schema
highlights_schema = {
    "type": "object",
    "properties": {
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["description", "time_frame"]
            }
        }
    }
}

# 7. Shopping Items Schema
shopping_schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["item", "time_frame"]
            }
        }
    }
}

# 8. Action Recognition Schema
actions_schema = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["action", "time_frame"]
            }
        }
    }
}

# 9. Sentiment Analysis Schema
sentiment_schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "examples": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# 10. Language Detection Schema
language_detection_schema = {
    "type": "object",
    "properties": {
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["language", "time_frame"]
            }
        }
    }
}

# 11. Keywords Schema
keywords_schema = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "importance_score": {"type": "number"}
                },
                "required": ["keyword", "importance_score"]
            }
        }
    }
}

# 12. OCR Schema
ocr_schema = {
    "type": "object",
    "properties": {
        "texts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["text", "time_frame"]
            }
        }
    }
}

# 13. Brand Logos Schema
brand_logos_schema = {
    "type": "object",
    "properties": {
        "logos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "brand": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["brand", "time_frame"]
            }
        }
    }
}

# 14. Social Media Summary Schema
social_media_summary_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"}
    }
}

# 15. Visual Style Schema
visual_style_schema = {
    "type": "object",
    "properties": {
        "style": {"type": "string"},
        "elements": {"type": "array", "items": {"type": "string"}}
    }
}

# 16. Anomalies Schema
anomalies_schema = {
    "type": "object",
    "properties": {
        "anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["description", "time_frame"]
            }
        }
    }
}

# 17. Gesture Recognition Schema
gestures_schema = {
    "type": "object",
    "properties": {
        "gestures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gesture": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["gesture", "time_frame"]
            }
        }
    }
}

# 18. Topics/Genres Schema
topics_schema = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "subtopics": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["topic"]
            }
        }
    }
}

# 19. Content Moderation Schema
moderation_schema = {
    "type": "object",
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "time_frame": {"type": "string"}
                },
                "required": ["type", "time_frame"]
            }
        }
    }
}

# 20. Contextual Ad Placement Schema
ad_placement_schema = {
    "type": "object",
    "properties": {
        "ads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "placement_time": {"type": "string"}
                },
                "required": ["description", "placement_time"]
            }
        }
    }
}

# Endpoints with improved prompts

@app.post("/analyze-video/tags", response_model=VideoAnalysisResponse)
async def generate_tags(request: VideoAnalysisRequest):
    prompt = (
        "Generate a list of relevant tags that best describe the main themes, objects, actions, and concepts in the video. "
        "Include a probability score for each tag's relevance. If there are no suitable tags, indicate that the content is general and untaggable."
    )
    return await analyze_video(request, prompt, tags_schema)

@app.post("/analyze-video/emotions", response_model=VideoAnalysisResponse)
async def detect_emotions(request: VideoAnalysisRequest):
    prompt = (
        "Analyze the video for visible emotions expressed by individuals through facial expressions and body language. "
        "List each emotion with a time frame and context where relevant. If no clear emotions are detected, state that the individuals are neutral or difficult to interpret."
    )
    return await analyze_video(request, prompt, emotions_schema)

@app.post("/analyze-video/description", response_model=VideoAnalysisResponse)
async def analyze_description(request: VideoAnalysisRequest):
    prompt = (
        "Generate a comprehensive description of the video's content, including main subjects, actions, visual elements, and tone. "
        "If the content is too abstract to describe, state that clearly."
    )
    return await analyze_video(request, prompt, description_schema)

@app.post("/analyze-video/scene-segmentation", response_model=VideoAnalysisResponse)
async def segment_scenes(request: VideoAnalysisRequest):
    prompt = (
        "Break down the video into distinct scenes with start and end times, and a brief description for each. "
        "If the video is one continuous scene, indicate that."
    )
    return await analyze_video(request, prompt, scene_segmentation_schema)

@app.post("/analyze-video/transcription", response_model=VideoAnalysisResponse)
async def transcribe_video(request: VideoAnalysisRequest):
    prompt = (
        "Provide a transcription of all spoken words in the video, with timestamps for each segment. "
        "If there is no spoken content, indicate this."
    )
    return await analyze_video(request, prompt, transcription_schema)

@app.post("/analyze-video/highlights", response_model=VideoAnalysisResponse)
async def summarize_highlights(request: VideoAnalysisRequest):
    prompt = (
        "Identify the key highlights of the video, with brief descriptions and time frames. "
        "If no particular highlights stand out, describe the video as evenly paced."
    )
    return await analyze_video(request, prompt, highlights_schema)

@app.post("/analyze-video/shopping", response_model=VideoAnalysisResponse)
async def identify_shopping_items(request: VideoAnalysisRequest):
    prompt = (
        "Identify any products, clothing, or accessories in the video that could be of interest for online shopping. "
        "Provide a brief description and time frame for each item. If no such items are present, indicate this."
    )
    return await analyze_video(request, prompt, shopping_schema)

@app.post("/analyze-video/actions", response_model=VideoAnalysisResponse)
async def recognize_actions(request: VideoAnalysisRequest):
    prompt = (
        "Describe the main actions or activities occurring in the video, with start and end timestamps for each. "
        "If the video is primarily static, state this."
    )
    return await analyze_video(request, prompt, actions_schema)

@app.post("/analyze-video/sentiment", response_model=VideoAnalysisResponse)
async def analyze_sentiment(request: VideoAnalysisRequest):
    prompt = (
        "Analyze the overall sentiment and emotional tone of the video, such as positive, negative, or neutral. "
        "If the sentiment is ambiguous or neutral, indicate this clearly."
    )
    return await analyze_video(request, prompt, sentiment_schema)

@app.post("/analyze-video/language-detection", response_model=VideoAnalysisResponse)
async def detect_language(request: VideoAnalysisRequest):
    prompt = (
        "Identify the languages spoken in the video, with time frames where each language is used. "
        "If no spoken language is detected, indicate that there is no audio or spoken content."
    )
    return await analyze_video(request, prompt, language_detection_schema)

@app.post("/analyze-video/keywords", response_model=VideoAnalysisResponse)
async def extract_keywords(request: VideoAnalysisRequest):
    prompt = (
        "Extract key phrases and keywords summarizing the video content. Provide an importance score and brief explanation for each. "
        "If there are no specific keywords, describe the video as generic or unremarkable in this regard."
    )
    return await analyze_video(request, prompt, keywords_schema)

@app.post("/analyze-video/ocr", response_model=VideoAnalysisResponse)
async def extract_text(request: VideoAnalysisRequest):
    prompt = (
        "Extract all visible text displayed within the video frames, with time frames where each text appears. "
        "If no text is visible, indicate that the video contains no readable text."
    )
    return await analyze_video(request, prompt, ocr_schema)

@app.post("/analyze-video/brand-logos", response_model=VideoAnalysisResponse)
async def detect_brand_logos(request: VideoAnalysisRequest):
    prompt = (
        "Identify any brand logos or trademarks appearing in the video, with time frames for each. "
        "If no logos are visible, indicate that the video does not contain identifiable brands."
    )
    return await analyze_video(request, prompt, brand_logos_schema)

@app.post("/analyze-video/social-media-summary", response_model=VideoAnalysisResponse)
async def social_media_summary(request: VideoAnalysisRequest):
    prompt = (
        "Generate a short, engaging summary of the video suitable for social media. "
        "If the content is abstract or challenging to summarize, indicate this."
    )
    return await analyze_video(request, prompt, social_media_summary_schema)

@app.post("/analyze-video/visual-style", response_model=VideoAnalysisResponse)
async def analyze_visual_style(request: VideoAnalysisRequest):
    prompt = (
        "Describe the visual style of the video, including color schemes, cinematography, and any notable visual effects. "
        "If the style is minimal or unremarkable, indicate this."
    )
    return await analyze_video(request, prompt, visual_style_schema)

@app.post("/analyze-video/anomalies", response_model=VideoAnalysisResponse)
async def detect_anomalies(request: VideoAnalysisRequest):
    prompt = (
        "Identify any unusual or unexpected events in the video, with time frames for each. "
        "If no anomalies are detected, state that the content is consistent."
    )
    return await analyze_video(request, prompt, anomalies_schema)

@app.post("/analyze-video/gestures", response_model=VideoAnalysisResponse)
async def recognize_gestures(request: VideoAnalysisRequest):
    prompt = (
        "Identify significant gestures made by individuals in the video, with descriptions and time frames. "
        "If no meaningful gestures are observed, indicate this."
    )
    return await analyze_video(request, prompt, gestures_schema)

@app.post("/analyze-video/topics", response_model=VideoAnalysisResponse)
async def identify_topics(request: VideoAnalysisRequest):
    prompt = (
        "Determine the main topics or genres covered in the video, with relevant subtopics where applicable. "
        "If no specific topics stand out, describe the content as general."
    )
    return await analyze_video(request, prompt, topics_schema)

@app.post("/analyze-video/moderation", response_model=VideoAnalysisResponse)
async def moderate_content(request: VideoAnalysisRequest):
    prompt = (
        "Analyze the video for any inappropriate, offensive, or restricted content, such as violence, nudity, or explicit language. "
        "If the video contains no such content, clearly state that it appears safe and appropriate."
    )
    return await analyze_video(request, prompt, moderation_schema)

# New Contextual Ad Placement Endpoint
@app.post("/analyze-video/contextual-ad-placement", response_model=VideoAnalysisResponse)
async def suggest_ad_placement(request: VideoAnalysisRequest):
    prompt = (
        "Suggest appropriate time frames for placing contextual ads in the video, with a brief context of the content at that time. "
        "If the video lacks suitable points for ad placement, indicate this clearly."
    )
    return await analyze_video(request, prompt, ad_placement_schema)

# Unified function to handle video analysis with schema
async def analyze_video(request: VideoAnalysisRequest, prompt: str, schema: dict) -> VideoAnalysisResponse:
    video_path = None
    try:
        _, _, video_path = download_and_extract_metadata(request.cdn_url)
        video_part = convert_video_to_part(video_path)
        result = generate_content(multimodal_model, prompt, video_part, schema)
        return VideoAnalysisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path:
            delete_temp_file(video_path)
