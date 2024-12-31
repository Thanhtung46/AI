import speech_recognition as sr

def transcribe_speech(lang='vi-VN'):
    """
    Hàm nhận diện giọng nói và chuyển thành văn bản.
    :param lang: Ngôn ngữ của giọng nói (mặc định là tiếng Việt).
    :return: Văn bản từ giọng nói.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Nói đi, tôi đang nghe...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language=lang)
            return text
        except sr.UnknownValueError:
            return "Không nhận diện được giọng nói."
        except sr.RequestError as e:
            return f"Lỗi khi kết nối dịch vụ: {e}"

