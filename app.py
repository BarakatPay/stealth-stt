import streamlit as st
import asyncio
import edge_tts
import tempfile
import os
import base64
import subprocess
import nest_asyncio
import time

nest_asyncio.apply()
st.set_page_config(page_title="ښه راغلاست", layout="wide")

# pashto_message = "ستړی مشې! دلته تاسو کولی شئ خپل سفر پیل کړئ."
pashto_message = """ښه راغلاست،..... 
مهرباني وکړئ انتظار وکړئ تر هغه چې موږ تاسو ته لارښوونه کوو."""
# 🔊 Generate TTS and return base64
async def generate_tts_b64(text: str) -> str:
    temp_path = tempfile.mktemp(suffix=".mp3")
    communicate = edge_tts.Communicate(text=text, voice="ps-AF-GulNawazNeural")
    await communicate.save(temp_path)

    with open(temp_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    os.remove(temp_path)

    return base64.b64encode(audio_bytes).decode("utf-8")

# 📦 Generate once to avoid delay on click
b64_audio = asyncio.get_event_loop().run_until_complete(generate_tts_b64(pashto_message))

# 🎉 Welcome Message
st.markdown(
    """
    <div style="text-align: center; margin-top: 100px;">
        <h1 style="font-size: 3.5rem;">ستړی مشې! 🌟</h1>
        <p style="font-size: 1.5rem;">دلته تاسو کولی شئ خپل سفر پیل کړئ</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 🎧 Audio tag (rendered only after click)
def render_audio(b64: str):
    st.markdown(
        f"""
        <audio controls autoplay style="display:None">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )

# 🚀 Launch app.py subprocess
def launch_new_app():
    subprocess.Popen(["streamlit", "run", "main.py"])

# 👇 Main button
if st.button("▶ غږ واورئ او دوام ورکړئ"):
    st.success("غږ پلی شو!")
    render_audio(b64_audio)
    # st.write("Loading app....")
    time.sleep(5.2)
    launch_new_app()
    # if st.button("➡ مخ ته لاړ شئ"):
    #     # st.success("نوی اپلیکیشن پیلېږي...")
    #     launch_new_app()
