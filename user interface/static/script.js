$(document).ready(function() {
    // Khi form được gửi
    $('#predict-form').on('submit', function(event) {
        event.preventDefault(); // Ngừng gửi form mặc định

        // Lấy giá trị từ các trường đầu vào
        const studentId = $('#student-id').val();
        const text = $('#text').val();

        // Gửi dữ liệu tới Flask API
        $.ajax({
            url: '/predict',  // Địa chỉ API Flask
            method: 'POST',
            data: {
                'student-id': studentId,  // ID của học sinh
                'text': text               // Tình trạng học sinh
            },
            success: function(response) {
                if (response.prediction !== undefined) {
                    // Hiển thị kết quả dự đoán
                    $('#prediction').text('Dự đoán: ' + (response.prediction === 0 ? 'Ổn định' : (response.prediction === 1 ? 'Căng thẳng' : 'Thất vọng')));
                    $('#trend-analysis').text('Phân tích xu hướng: ' + response.trend_analysis);
                    $('.result').show();  // Hiển thị phần kết quả
                } else if (response.error) {
                    alert('Lỗi: ' + response.error);  // Nếu có lỗi
                }
            },
            error: function(xhr, status, error) {
                alert('Đã có lỗi xảy ra: ' + error);  // Nếu có lỗi trong yêu cầu
            }
        });
    });

    // Xóa kết quả cũ khi người dùng thay đổi thông tin
    $('#student-id').on('input', function() {
        $('.result').hide(); // Ẩn kết quả khi người dùng thay đổi mã học sinh
    });

    $('#text').on('input', function() {
        $('.result').hide(); // Ẩn kết quả khi người dùng thay đổi tình trạng
    });
});
