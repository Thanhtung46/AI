from PIL import Image
import pytesseract

def extract_text_from_image(image_path, lang='vie'):
    """
    Hàm sử dụng OCR để quét văn bản từ ảnh.
    :param image_path: Đường dẫn đến ảnh.
    :param lang: Ngôn ngữ của văn bản ('vie' cho tiếng Việt).
    :return: Văn bản trích xuất được từ ảnh.
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
        return text.strip()
    except Exception as e:
        return f"Lỗi khi xử lý ảnh: {e}"
