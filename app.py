from fastapi import FastAPI, File, UploadFile
import whisper
import tempfile

app = FastAPI()
#model = whisper.load_model("base")
model = whisper.load_model("medium")


#@app.post("/transcribe")
#async def transcribe(file: UploadFile = File(...)):
#    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
#        temp_audio.write(await file.read())
#        temp_audio_path = temp_audio.name
#    result = model.transcribe(temp_audio_path)
#    return {"text": result["text"]}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    try:
        result = model.transcribe(
            temp_audio_path,
            language="ru",       # Язык для повышения точности
            temperature=0.0,     # Низкая вариативность — стабильность распознавания
            beam_size=5,         # Расширенный поиск гипотез
            best_of=5,           # Выбор лучшей гипотезы из нескольких
            task="transcribe"    # Явная транскрипция (не перевод)
        )
    finally:
        os.remove(temp_audio_path)  # Удаляем временный файл

    return {"text": result["text"]}
