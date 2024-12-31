from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Chào mừng bạn đến với ứng dụng Flask!"

if __name__ == '__main__':
    app.run(debug=True)