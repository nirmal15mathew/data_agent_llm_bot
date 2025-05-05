from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
socketio = SocketIO(app)

def get_bot_response(prompt):
    time.sleep(1)
    return f"Echo: {prompt}"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_message')
def handle_user_message(data):
    response = get_bot_response(data['prompt'])
    emit('bot_response', {'response': response})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
