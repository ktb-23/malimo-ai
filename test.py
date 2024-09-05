import os  # os 모듈 추가
from flask import Flask, request, jsonify, Response
from openai import OpenAI
import logging
from logging.handlers import RotatingFileHandler
import json

# 환경 변수에서 OpenAI API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Flask 앱 초기화
app = Flask(__name__)

# 로깅 설정
handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# 사용자 메시지 처리 함수
def handle_user_message(session_id, user_input):
    messages = [
        {"role": "system", "content": "너는 사용자의 일기로부터 감정을 읽어내고 사용자를 위로해주는 역할이야."},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        app.logger.info(f"GPT response: {answer}")

        return answer
    except Exception as e:
        app.logger.error(f"Error during OpenAI API call: {e}")
        return "죄송합니다, 응답 처리 중 오류가 발생했습니다."

@app.route('/review', methods=['POST'])
def review():
    data = request.get_json()

    user_input = data.get('text')
    session_id = data.get('session_id', 'default')
    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    try:
        bot_response = handle_user_message(session_id, user_input)
        return Response(json.dumps({"responseText": bot_response}, ensure_ascii=False), content_type='application/json; charset=utf-8')
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
