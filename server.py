import os
import sys
from flask import Flask, request, jsonify, Response
from openai import OpenAI
import json
import traceback
import re
import logging

# 환경 변수에서 OpenAI API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

client = OpenAI(api_key=api_key)

# Flask 앱 초기화
app = Flask(__name__)

# 콘솔 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

@app.route('/get_or_create_assistant', methods=['POST'])
def get_or_create_assistant():
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({"error": "No user_id provided"}), 400
        
        user_id = data['user_id']
        
        logger.debug(f"Creating assistant for user {user_id}")
        assistant = client.beta.assistants.create(
            name=f"Diary Assistant for {user_id}",
            instructions="""
너는 사용자의 일기를 분석하는 역할이야. 그리고 이 데이터를 구조화된 JSON 형식으로 제공해야 해. 아래와 같은 형식으로 응답을 생성해야 해:

{
  "emotion_analysis": [
    {"emotion": "피곤함", "percentage": 40},
    {"emotion": "기쁨", "percentage": 30},
    {"emotion": "불안", "percentage": 20},
    {"emotion": "기대", "percentage": 10}
  ],
  "total_score": 3.5,
  "summary": [
    "오늘 일하다가 너무 힘들어 퇴근하기로 결정했어요.",
    "저녁에 친구와 맛있는 저녁을 먹었어요.",
    "집에 돌아와서 영화를 보며 휴식을 취했어요."
  ],
  "advice": "오늘 하루 정말 고생 많으셨습니다. 힘들었던 일이 많았지만, 내일은 더 나은 하루가 될 거예요. 충분한 휴식을 취하고 긍정적인 마음을 유지하려고 노력해보세요. 새로운 도전은 때때로 불안함을 가져오기도 하지만, 그 과정에서 배울 점이 많습니다. 자신을 믿고 조금 더 여유를 가지면 좋을 것 같아요."
}

- "emotion_analysis"는 각 감정과 해당 감정의 비율(%)을 나타내는 객체 배열이야.
- "total_score"는 오늘 하루의 총점을 나타내며, 0에서 5 사이의 숫자 값을 가질 수 있어.
- "summary"는 3문장으로 사용자의 하루를 요약한 리스트야.
- "advice"는 사용자에게 따뜻한 위로나 조언을 제공하는 텍스트야.
- 모든 응답은 JSON 형식으로 반환되어야 해.
""",
            model="gpt-4o-mini"
        )
        
        thread = client.beta.threads.create()
        logger.info(f"Assistant created for user {user_id}: assistant_id={assistant.id}, thread_id={thread.id}")
        return jsonify({"assistant_id": assistant.id, "thread_id": thread.id}), 200
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(f"Error in create_assistant: {error_message}\n{traceback.format_exc()}")
        return jsonify({"error": error_message}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'thread_id' not in data or 'message' not in data or 'assistant_id' not in data:
            return jsonify({"error": "Invalid request data"}), 400
        
        thread_id = data['thread_id']
        user_input = data['message']
        assistant_id = data['assistant_id']
        
        logger.info(f"Analyzing message for thread ID: {thread_id}, assistant ID: {assistant_id}")
        
        analysis_result = analyze_message(thread_id, user_input, assistant_id)
        return Response(json.dumps(analysis_result, ensure_ascii=False), content_type='application/json; charset=utf-8')
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(f"Error in analyze: {error_message}\n{traceback.format_exc()}")
        return jsonify({"error": error_message}), 500

def analyze_message(thread_id, user_input, assistant_id):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        while run.status != 'completed':
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            logger.debug(f"Run status: {run.status}")
        
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        answer = messages.data[0].content[0].text.value
        
        logger.debug(f"Raw API response: {answer}")
        
        parsed_response = gpt_response(answer)
        
        if all(value == "오류" for value in parsed_response.values()):
            logger.error(f"Parsing failed for all fields. Raw response: {answer}")
            return {"error": "응답 중 오류가 발생했습니다."}
        
        return parsed_response
    except Exception as e:
        logger.error(f"Error analyzing message: {str(e)}\n{traceback.format_exc()}")
        raise

import re
import logging

# logger 설정
logger = logging.getLogger(__name__)

def gpt_response(response):
    try:
        logger.debug(f"Parsing response: {response}")

        # JSON 데이터로 처리
        if not isinstance(response, dict):
            raise ValueError("Response is not a valid JSON object")

        # 각 항목 추출
        emotion_analysis = response.get("emotion_analysis", "감정 분석을 찾을 수 없습니다.")
        total_score = response.get("total_score", "총점을 찾을 수 없습니다.")
        summary = response.get("summary", "요약을 찾을 수 없습니다.")
        advice = response.get("advice", "조언을 찾을 수 없습니다.")

        # 결과 딕셔너리
        parsed = {
            "emotion_analysis": emotion_analysis,
            "total_score": total_score,
            "summary": summary,
            "advice": advice
        }
        logger.debug(f"Parsed response: {parsed}")
        return parsed
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}\nResponse: {response}")
        return {
            "emotion_analysis": "오류",
            "total_score": "오류",
            "summary": "오류",
            "advice": "오류"
        }
