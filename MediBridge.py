import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langdetect import detect
from googletrans import Translator
from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Function to get video ID from YouTube URL
def get_video_id(video_url):
    try:
        query = urlparse(video_url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                return parse_qs(query.query)['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
    except Exception as e:
        st.error(f"Error extracting video ID: {e}")
    return None

# Function to transcribe YouTube video, translate, and summarize
def process_youtube_video(video_url):
    video_id = get_video_id(video_url)

    if video_id:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcriptions = [entry['text'] for entry in transcript]

            st.subheader("Transcribed Text:")
            st.write(' '.join(transcriptions))  # Display the transcribed text

            # Concatenate all transcriptions into a single string
            text = ' '.join(transcriptions)
            language = detect(text)

            if language != 'en':
                translator = Translator()
                text = translator.translate(text, src=language, dest='en').text

            # Summarize using the BART model
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs, max_length=50, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
            summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.subheader("Summarized Text:")
            st.write(summarized_text)  # Display the summarized text

            return summarized_text
        except Exception as e:
            st.error(f"Error processing YouTube video: {e}")
            return None
    else:
        st.error("Failed to extract video ID.")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Imaginatrix", layout="wide", initial_sidebar_state="expanded")

    st.header("MediBridge", divider='rainbow')
    st.subheader("Bridging Language Gaps in Surgical Videos through Translation and Summarization")
    st.write("Type 'exit' to end the conversation.")

    user_input = st.text_input("Enter YouTube video URL:")
    if user_input.lower() == 'exit':
        st.write("Exiting the conversation. Goodbye!")
    elif user_input:
        st.write("Processing... Please wait.")
        process_youtube_video(user_input)

if __name__ == "__main__":
    main()
