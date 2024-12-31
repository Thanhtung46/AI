document.getElementById('predictForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Ngừng việc gửi form mặc định
    
    // Lấy dữ liệu từ form
    const formData = new FormData(event.target);
    
    // Chuyển đổi formData thành một đối tượng có thể sử dụng được trong JSON
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    
    // Gửi yêu cầu POST đến Flask để nhận dự đoán
    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams(data)
    })
    .then(response => response.json())
    .then(result => {
        // Hiển thị kết quả dự đoán
        if (result.prediction !== undefined) {
            document.getElementById('predictionResult').innerText = `Dự đoán: ${result.prediction}`;
        } else if (result.error) {
            document.getElementById('predictionResult').innerText = `Lỗi: ${result.error}`;
        }
    })
    .catch(error => {
        console.error('Có lỗi xảy ra:', error);
    });
});
