import subprocess
import sys

required_packages = [
    "groq",
    "gradio",
    "gTTS",
    "requests",
    "regex",
    "jieba",
    "PyYAML"  # PyYAML provides the yaml module
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in required_packages:
    try:
        # Special handling for PyYAML which is imported as 'yaml'
        if package == "PyYAML":
            import yaml
        else:
            __import__(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        install(package)

# Your script's imports
import gradio as gr
from gtts import gTTS
import requests
import tempfile
import os
import difflib
import regex
import sys
import jieba
import yaml
import random

# Mapping of user-friendly language names to gTTS and Groq Whisper API language codes
LANGUAGE_CODES = {
    'English': {'gtts': 'en', 'whisper': 'en'},
    'Spanish': {'gtts': 'es', 'whisper': 'es'},
    'Chinese': {'gtts': 'zh', 'whisper': 'zh'},
    'Russian': {'gtts': 'ru', 'whisper': 'ru'}
}

# Your Groq API Key (Replace this with your actual API key)
GROQ_API_KEY = 'gsk_xxxxxxxxxxxxxxxxxxxxxxxxxx'

def generate_tts(language, poem_text, line_range):
    """
    Generates TTS audio for the selected lines of the poem.

    Parameters:
    - language (str): Selected language.
    - poem_text (str): The full poem text.
    - line_range (tuple): Tuple containing start and end line numbers.

    Returns:
    - audio_file (temp file): Generated TTS audio file.
    - error (str): Error message if any.
    """
    try:
        # Split the poem into individual lines
        lines = poem_text.strip().split('\n')
        total_lines = len(lines)

        # Extract start and end lines from the range
        start, end = line_range

        # Validate the line range
        if total_lines < end:
            end = total_lines
        if start > end:
            start = end

        # Select the specified lines
        selected_lines = lines[start-1:end]
        tts_text = '\n'.join(selected_lines)

        # Initialize gTTS with the appropriate language code
        tts = gTTS(text=tts_text, lang=LANGUAGE_CODES[language]['gtts'])

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_file = fp.name
            tts.save(temp_file)

        return temp_file, None

    except Exception as e:
        return None, str(e)

def preprocess(text, language=''):
    """
    Remove special characters and punctuation, normalize whitespace,
    and convert text to lowercase. Supports multiple languages.
    """
    if language.lower() == 'chinese':
        # Perform word segmentation for Chinese text using jieba
        tokens = jieba.cut(text)
        text = ' '.join(tokens)

    # Normalize whitespace characters (space, tab, newline) to a single space
    text = regex.sub(r'\s+', ' ', text)

    # Remove punctuation and special characters
    # This regex retains Unicode letters and numbers
    text = regex.sub(r'[^\p{L}\p{N}\s]', '', text)

    # Convert to lowercase for case-insensitive comparison (irrelevant for Chinese)
    text = text.lower()

    return text.strip()

def transcribe_and_score(audio_path, language, poem_text):
    """
    Transcribes the user's audio using Groq Whisper API and computes a similarity score.
    
    Parameters:
    - audio_path (str): Path to the user's audio file.
    - language (str): Selected language.
    - poem_text (str): The full poem text.
    
    Returns:
    - transcription (str): Transcribed text from the audio.
    - score (str): Similarity score as a percentage.
    """
    try:
        if audio_path is None:
            return "No audio provided.", "0%"

        # Initialize the Groq client with your API key
        client = Groq(api_key=GROQ_API_KEY)

        # Specify the path to the audio file
        filename = audio_path  # Replace with your audio file path

        # Open the audio file
        with open(filename, "rb") as file:
            # Create a transcription of the audio file
            transcription_response = client.audio.transcriptions.create(
                file=(filename, file.read()),  # Required audio file
                model="whisper-large-v3-turbo",  # Required model to use for transcription
                language=LANGUAGE_CODES[language]['whisper'],  # Optional
                response_format="json",  # Optional, default is json
                temperature=0.0  # Optional
            )
            
            # Extract the transcription text
            transcription = transcription_response.text.strip()

            if not transcription:
                return "Transcription failed.", "0%"

            # Preprocess the strings
            clean_str1 = preprocess(poem_text, language)
            clean_str2 = preprocess(transcription, language)
            # Create a SequenceMatcher object
            matcher = difflib.SequenceMatcher(None, clean_str1, clean_str2)
            # Get the similarity ratio and convert to percentage
            similarity = round((matcher.ratio() * 100)**2, 0)

            return transcription, f"{similarity}%"

    except Exception as e:
        return f"An error occurred: {e}", "0%"

### Mathematics functions

def generate_question():
    operations = ['*', '/', '+', '-']
    operation = random.choice(operations)
    if operation in ['+', '-']:
        num1 = random.randint(100, 999)
        num2 = random.randint(1000, 9999)
        if operation == '-':
            num1, num2 = sorted((num1, num2), reverse=True)  # Ensure positive result
        answer = num1 + num2 if operation == '+' else num1 - num2
    elif operation in ['*', '/']:
        num1 = random.randint(10, 99)
        num2 = random.randint(100, 999)
        if operation == '/':
            while True:
                num1, num2 = sorted((random.randint(100, 999), random.randint(10, 99)), reverse=True)
                if num1 % num2 == 0 and num1 // num2 > 1:
                    answer = num1 // num2
                    break
        else:
            answer = num1 * num2
    return f"{num1} {operation} {num2}", answer

def open_config():
    if not os.path.exists("/root/config_homework.yaml"):
        current_question, correct_answer = generate_question()
        default_config = {
            "correct_increment": 1,
            "wrong_decrement": 0.2,
            "target_points": 100,
            "current_question": current_question,
            "correct_answer": correct_answer,
            "user_points": 0
        }
        with open("/root/config_homework.yaml", "w") as file:
            yaml.dump(default_config, file)
    with open("/root/config_homework.yaml", "r") as file:
        return yaml.safe_load(file)

def save_config(config):
    with open("/root/config_homework.yaml", "w") as file:
        yaml.dump(config, file)

def process_input(user_input):
    config = open_config()
    try:
        user_input = float(user_input)
        if abs(user_input - config['correct_answer']) < 0.1:
            config['user_points'] += config['correct_increment']
            message = f"Correct! Your points: {config['user_points']:.1f}"
        else:
            config['user_points'] -= config['wrong_decrement']
            message = f"Wrong! The correct answer was {config['correct_answer']}. Your points: {config['user_points']:.1f}"
        
        # Check if user has reached the target points
        if config['user_points'] >= config['target_points']:
            save_config(config)
            return "You have finished! Congratulations!", None, gr.update(value="")
        else:
            # Generate a new question and update the config
            new_question, new_answer = generate_question()
            config['current_question'] = new_question
            config['correct_answer'] = new_answer
            save_config(config)
            return message, new_question, gr.update(value="")
    except ValueError:
        # Handle invalid input
        new_question, new_answer = generate_question()
        config['current_question'] = new_question
        config['correct_answer'] = new_answer
        save_config(config)
        return (
            f"Invalid input. The correct answer was {config['correct_answer']}. \nYour points: {config['user_points']:.1f}",
            new_question,
            gr.update(value="")
        )


def main():

    with gr.Blocks() as demo:
        with gr.Tab("📜 Poem Learning 📜"):
            # Initialize state to store language and poem
            state = gr.State(value={"language": "English", "poem": ""})
            
            with gr.Column():
                # Language selection dropdown
                language_dropdown = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value='English',
                    label='Select Language'
                )
                
                # Poem text input
                poem_input = gr.Textbox(
                    lines=6,
                    placeholder='Enter your poem here with at least 6 lines.',
                    label='Poem'
                )
            
            # Define the update functions inside main to access 'state'
            def update_state_on_language_change(language, current_state):
                """
                Updates the state with the new language while preserving the poem.
                """
                current_poem = current_state.get("poem", "")
                new_state = {"language": language, "poem": current_poem}
                return new_state
            
            def update_state_on_poem_change(new_poem, current_state):
                """
                Updates the state with the new poem while preserving the language.
                """
                current_language = current_state.get("language", "")
                new_state = {"language": current_language, "poem": new_poem}
                return new_state
            
            # Connect the dropdown change to update the state
            language_dropdown.change(
                fn=update_state_on_language_change,
                inputs=[language_dropdown, state],
                outputs=state
            )
            
            # Connect the poem input change to update the state
            poem_input.change(
                fn=update_state_on_poem_change,
                inputs=[poem_input, state],
                outputs=state
            )
            
            with gr.Tab("Practice"):
                with gr.Column():
                    # Line range selectors
                    with gr.Row():
                        start_line = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=1,
                            label='Start Line'
                        )
                        end_line = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=6,
                            label='End Line'
                        )
                    
                    # Generate TTS button and audio output
                    generate_tts_btn = gr.Button("Generate TTS")
                    tts_output = gr.Audio(label="Generated Audio")
                    
                    # Error message output
                    tts_error = gr.Textbox(label="Error Message", visible=False, interactive=False)
                    
                    # Function to handle TTS generation and update state
                    def handle_tts(language, poem, start, end, current_state):
                        lines = poem.strip().split('\n')
                        total_lines = len(lines)

                        audio_path, error = generate_tts(language, poem, (int(start), int(end)))
                        if error:
                            return gr.Audio(value=None), gr.Textbox(value=error, visible=True), gr.Slider(maximum=total_lines), gr.Slider(maximum=total_lines)
                        else:
                            # Update state with current language and poem
                            new_state = {"language": language, "poem": poem}
                            return audio_path, gr.Textbox(value="", visible=False), gr.Slider(maximum=total_lines), gr.Slider(maximum=total_lines)
                    
                    # Connect the button to the TTS function
                    generate_tts_btn.click(
                        fn=handle_tts,
                        inputs=[language_dropdown, poem_input, start_line, end_line, state],
                        outputs=[tts_output, tts_error, start_line, end_line]
                    )
            
            with gr.Tab("Trial"):
                with gr.Column():
                    # Audio input for user's recitation
                    user_audio = gr.Audio(sources=["microphone"], type="filepath", label="Record Your Recitation")
                    
                    # Send button
                    send_btn = gr.Button("Send")
                    
                    # Outputs for transcription and score
                    transcription_output = gr.Textbox(label="Transcription", interactive=False)
                    score_output = gr.Textbox(label="Similarity Score", interactive=False)
                    
                    # Function to handle transcription and scoring
                    def handle_trial(audio, current_state):
                        try:
                            transcription, score = transcribe_and_score(audio, current_state['language'], current_state['poem'])
                            return transcription, score
                        except:
                            return "", 0
                    
                    # Connect the button to the trial function
                    send_btn.click(
                        fn=handle_trial,
                        inputs=[user_audio, state],
                        outputs=[transcription_output, score_output]
                    )
                    
                    # Information about scoring
                    gr.Markdown("### 📊 Score Interpretation")
                    gr.Markdown("""
                    - **0%:** No similarity between your recitation and the poem.
                    - **100%:** Perfect match with the poem.
                    - **Intermediate Values:** Reflect the degree of similarity.
                    """)
            
            # Footer with instructions
            gr.Markdown("""
            ---
            **Instructions:**
            1. Navigate to the **Practice** tab.
            2. Select a language and input a poem with at least six lines.
            3. Choose the range of lines you want to practice.
            4. Click **Generate TTS** to hear the selected lines.
            5. Switch to the **Trial** tab to record your recitation.
            6. Click **Send** to transcribe and score your recitation.
            """)
    
        with gr.Tab("👩‍🔬 Mathematics Learning 🧮"):
            points_markdown = gr.Markdown(f"Points: {open_config()['user_points']:.1f}")
            current_label = gr.Markdown(open_config()['current_question'])
            answer_input = gr.Textbox(label="Your answer")
            submit_button = gr.Button("Submit")
            submit_button.click(
                process_input, 
                inputs=[answer_input], 
                outputs=[points_markdown, current_label, answer_input]
            )

    # Launch the Gradio app
    # demo.launch(server_name="0.0.0.0", server_port=666)
    demo.launch()

if __name__ == "__main__":
    main()
