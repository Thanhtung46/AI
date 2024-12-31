def generate_dynamic_recommendations(prediction, text):
    """
    Tạo gợi ý tâm lý dựa trên nhãn dự đoán và ngữ cảnh văn bản.
    :param prediction: Nhãn dự đoán (0: bình thường, 1: căng thẳng, 2: thất vọng).
    :param text: Văn bản liên quan.
    :return: Gợi ý cho giáo viên.
    """
    base_recommendations = {
        0: "Học sinh bình thường, cần khuyến khích động viên.",
        1: "Học sinh có dấu hiệu căng thẳng. Nên trao đổi trực tiếp và tạo không gian để chia sẻ.",
        2: "Học sinh thất vọng, cần hỗ trợ tâm lý chuyên sâu hoặc kết nối với chuyên gia.",
    }

    # Gợi ý thêm dựa trên từ khóa
    additional_recommendations = []
    if "khóc" in text:
        additional_recommendations.append("Học sinh có dấu hiệu cảm xúc tiêu cực. Nên hỏi thăm và động viên.")
    if "ngủ" in text:
        additional_recommendations.append("Học sinh có dấu hiệu mệt mỏi. Hỏi về lịch sinh hoạt và tình trạng sức khỏe.")

    final_recommendation = base_recommendations.get(prediction, "Không rõ trạng thái tâm lý.")
    if additional_recommendations:
        final_recommendation += " " + " ".join(additional_recommendations)

    return final_recommendation
