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
  "summary": "오늘 일하다가 너무 힘들어 퇴근하기로 결정했어요. 저녁에 친구와 맛있는 저녁을 먹었어요. 집에 돌아와서 영화를 보며 휴식을 취했어요.",
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
        # 사용자의 메시지를 thread에 추가
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )
        
        # 새로운 run 생성
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        # run이 완료될 때까지 대기
        while run.status != 'completed':
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            logger.debug(f"Run status: {run.status}")
        
        # 메시지 리스트를 받아서 GPT의 응답을 추출
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        
        # 디버깅용으로 전체 메시지 출력
        logger.debug(f"Messages received: {messages}")

        # role이 'assistant'인 메시지를 필터링하여 찾기
        assistant_message = None
        for message in messages.data:
            if message.role == 'assistant':
                assistant_message = message.content[0].text.value
                break

        if assistant_message is None:
            logger.error("No assistant message found in the thread.")
            return {"error": "No assistant message found."}

        logger.debug(f"Assistant message extracted: {assistant_message}")
        
        # 응답이 JSON 형식이므로 이를 파싱
        try:
            parsed_response = json.loads(assistant_message)
        except json.JSONDecodeError:
            logger.error(f"API 응답이 JSON 형식이 아님: {assistant_message}")
            return {"error": "API 응답이 JSON 형식이 아닙니다."}
        
        # 각 항목이 제대로 존재하는지 확인
        required_keys = ["emotion_analysis", "total_score", "summary", "advice"]
        if all(key in parsed_response for key in required_keys):
            logger.debug(f"Parsed response: {parsed_response}")

            # emotion_analysis를 문자열로 변환
            emotion_analysis = parsed_response["emotion_analysis"]
            emotion_analysis_str = ", ".join([f"{item['emotion']}: {item['percentage']}%" for item in emotion_analysis])

            # 최종 결과
            parsed_result = {
                "emotion_analysis": emotion_analysis_str,
                "total_score": parsed_response["total_score"],
                "summary": parsed_response["summary"],
                "advice": parsed_response["advice"]
            }

            logger.debug(f"Final parsed result: {parsed_result}")
            return parsed_result
        else:
            logger.error(f"응답에 필수 항목 누락: {parsed_response}")
            return {"error": "응답에 필수 항목이 누락되었습니다."}

    except Exception as e:
        logger.error(f"Error analyzing message: {str(e)}\n{traceback.format_exc()}")
        raise

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
        result = {
            "emotion_analysis": emotion_analysis,
            "total_score": total_score,
            "summary": summary,
            "advice": advice
        }
        logger.debug(f"Parsed response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}\nResponse: {response}")
        return {
            "emotion_analysis": "오류",
            "total_score": "오류",
            "summary": "오류",
            "advice": "오류"
        }
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)