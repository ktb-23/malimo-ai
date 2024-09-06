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

            # summary 리스트를 하나의 문자열로 변환
            summary_list = parsed_response["summary"]
            summary_str = " ".join(summary_list)  # 각 문장을 공백으로 연결

            # 최종 결과
            parsed_result = {
                "emotion_analysis": emotion_analysis_str,
                "total_score": parsed_response["total_score"],
                "summary": summary_str,  # 리스트를 문자열로 변환한 값
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