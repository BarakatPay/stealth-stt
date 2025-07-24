import asyncio
import tempfile
import streamlit as st
import torch
from pydub import AudioSegment
import edge_tts
import nest_asyncio
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from groq import Groq
import numpy as np
import time

# 🛠 Load .env
load_dotenv()

# 🧠 Config
HF_MODEL_ID = st.secrets["HF_MODEL_ID"]
TTS_VOICE = st.secrets["TTS_VOICE"]
TARGET_SR = int(st.secrets["TARGET_SR"])
HF_TOKEN = st.secrets["HF_API_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not HF_TOKEN or not GROQ_API_KEY:
    st.error("❌ Missing HF_TOKEN or GROQ_API_KEY in your .env file.")
    st.stop()

# 🔐 Hugging Face login
login(token=HF_TOKEN)

# 🧠 Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# 📄 Streamlit UI
nest_asyncio.apply()
st.set_page_config(page_title="Pashto STT → Groq → Pashto TTS", page_icon="🎙", layout="centered")
st.title("🎙 Pashto STT → LLM → Pashto TTS")

st.subheader("🎤 Record Pashto Audio")
audio_value = st.audio_input("🎙 Record a voice message")
uploaded_file = False


@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(HF_MODEL_ID)
    model.eval()
    return processor, model


processor, model = load_model()


async def generate_pashto_tts(text: str, output_path: str):
    communicator = edge_tts.Communicate(text, voice=TTS_VOICE)
    await communicator.save(output_path)


def generate_tts_blocking(text: str, path: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_pashto_tts(text, path))


def convert_to_wav(uploaded_bytes: bytes, ext: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_in:
        temp_in.write(uploaded_bytes)
        input_path = temp_in.name
    output_path = input_path.replace(f".{ext}", ".wav")
    audio = AudioSegment.from_file(input_path, format=ext)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    audio.export(output_path, format="wav")
    return output_path


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # normalize
    return torch.from_numpy(samples)


if audio_value or uploaded_file:
    if audio_value:
        audio_bytes = audio_value.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_wav_path = f.name
    elif uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        audio_bytes = uploaded_file.read()
        if ext != "wav":
            temp_wav_path = convert_to_wav(audio_bytes, ext)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                temp_wav_path = f.name

    # st.audio(audio_bytes, format="audio/wav")

    try:
        waveform = load_audio(temp_wav_path)
    except Exception as e:
        st.error(f"❌ Failed to load audio: {e}")
        st.stop()

    # st.subheader("📈 Audio Waveform")
    # downsampled = waveform.numpy()[::10]
    # st.line_chart(downsampled)
    # st.markdown(f"⏱ **Duration**: {waveform.shape[-1] / TARGET_SR:.2f} sec")

    inputs = processor(waveform, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    st.subheader("📝 Transcription")
    st.success(transcription)

    prompt = f"""
              ستا دنده دا ده چې کاروونکو ته په پښتو ژبه لنډ، دقیق او دوستانه ځوابونه ورکړې، لکه یو ښه ملګری.
              [Your task is to provide concise, accurate, and friendly answers to users in Pashto, like a good friend.]
    
              بېلګې:
              [Examples:]
    
              User says (in Pashto): سلام، ته څنګه يې؟
              Model reply (in Pashto): زه ښه يم، مننه. ته څنګه يې؟
              [User: Hello, how are you? -> Model: I am fine, thank you. How are you?]
    
              User says (in Pashto): د کابل هوا څنګه ده؟
              Model reply (in Pashto): نن په کابل کې هوا وریځ ده. ښه به وي که چترۍ درسره وي!
              [User: How is the weather in Kabul? -> Model: The weather in Kabul today is cloudy. It would be good to have an umbrella with you!]
    
              User says (in Pashto): د پښتنو تاریخ څه دی؟
              Model reply (in Pashto): پښتانه یوه لرغونې قوم ده چې بډایه کلتور او تاریخ لري. ډېر په زړه پورې!
              [User: What is the history of Pashtuns? -> Model: Pashtuns are an ancient people with a rich culture and history. Very interesting!]
    
              User says (in Pashto): تر ټولو لوړ غر کوم دی؟
              Model reply (in Pashto): ایوریسټ! ایا ته غواړې چې نور معلومات هم درکړم؟
              [User: What is the highest mountain? -> Model: Everest! Would you like me to give you more information?]
    
              User says (in Pashto): افغانستان د کومې قارې برخه دی؟
              Model reply (in Pashto): اسیا. آیا نور څه هم غواړې چې پوه شې؟
              [User: Which continent is Afghanistan part of? -> Model: Asia. Is there anything else you'd like to know?]
    
              User says (in Pashto): د موټر چلوولو لپاره څه ته اړتیا لرو؟
              Model reply (in Pashto): موټر چلوونکی جواز ته اړتیا لرې. ته غواړې موټر وچلوې؟
              [User: What do we need to drive a car? -> Model: You need a driving license. Do you want to drive a car?]
    
              User says (in Pashto): اوبه له کومو عناصرو جوړې دي؟
              Model reply (in Pashto): اوبه له هایدروجن او اکسیجن څخه جوړې دي. حیرانوونکې نه ده؟
              [User: What elements are water made of? -> Model: Water is made of Hydrogen and Oxygen. Isn't that amazing?]
    
              User says (in Pashto): {transcription}.
              Model reply (in Pashto):
            """

    try:
        with st.spinner("🤖 Thinking..."):
            groq_response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful Pashto-speaking assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile"
            )
        pashto_reply = groq_response.choices[0].message.content.strip()
    except Exception as e:
        pashto_reply = f"❌ Error: {e}"

    st.subheader("🤖 LLM says")
    st.info(pashto_reply)

    st.subheader("🔊 Pashto TTS")
    tts_out = "groq_pashto.mp3"
    try:
        generate_tts_blocking(pashto_reply, tts_out)
        with open(tts_out, "rb") as f:
            tts_bytes = f.read()
        st.audio(tts_bytes, format="audio/mp3")
        # st.download_button("⬇ Download Audio", tts_bytes, file_name="groq_pashto.mp3")
    except Exception as e:
        st.error(f"TTS Generation Failed: {e}")
    finally:
        time.sleep(5)
        st.empty()
else:
    st.warning("📢 Speak to begin.")
    st.empty()
