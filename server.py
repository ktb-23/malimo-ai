import os
from flask import Flask, request, jsonify, Response
from openai import OpenAI
import logging
from logging.handlers import RotatingFileHandler
import json
import traceback
import re
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv()

# 환경 변수에서 OpenAI API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

client = OpenAI(api_key=api_key)

# 환경 변수에서 OpenAI API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

client = OpenAI(api_key=api_key)

# Flask 앱 초기화
app = Flask(__name__)

# 로깅 설정
handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# 사용자별 assistant 관리를 위한 딕셔너리
user_assistants = {}

def create_assistant(user_id):
    try:
        app.logger.debug(f"Creating assistant for user {user_id}")
        assistant = client.beta.assistants.create(
            name=f"Diary Assistant for {user_id}",
            instructions="""너는 사용자의 일기를 분석하는 역할이야. 다음 세 가지 정보를 제공해야 해:

1. 감정 분석:
   - 주요 감정들을 구체적으로 나열하고 각각의 비율(%)을 제시해. (예: 피곤함: 40%, 기쁨: 30%, 불안: 20%, 기대: 10%)
   - 감정은 '부정', '긍정', '중립' 등의 일반적인 카테고리가 아닌, 구체적인 감정 단어를 사용해야 해.
   - 최소 3개, 최대 5개의 감정을 나열해.
   - 반드시 오늘 하루의 총점(0-5점, 0.5점 단위)을 제시해. (예: 총점: 3.5/5)

2. 요약:
   - 사용자의 하루를 객관적으로 요약해줘.
   - 정확히 3문장으로 작성해.
   - 각 문장은 반드시 '-했어요' 어미로 끝나야 해. (예: "오늘 일하다가 너무 힘들어 퇴근하기로 결정했어요.")
   - '-'나 다른 기호 없이 문장만 작성해.

3. 조언:
   - 사용자의 감정 상태에 맞는 따뜻한 위로나 조언을 제공해.
   - 100-200단어 정도로 작성해.

반드시 위의 순서와 형식을 지켜서 답변해줘. 각 섹션 시작 시 번호와 제목을 명확히 표시해.""",
            model="gpt-4o-mini"
        )
        app.logger.debug(f"Assistant created successfully for user {user_id}")
        return assistant.id
    except Exception as e:
        app.logger.error(f"Error creating assistant for user {user_id}: {str(e)}")
        raise

def parse_response(response):
    try:
        app.logger.debug(f"Parsing response: {response}")
        
        # 정규 표현식을 사용하여 각 섹션을 찾습니다.
        emotion_pattern = r"1\.?\s*감정\s*분석:?([\s\S]*?)(?=2\.|\n\n|$)"
        summary_pattern = r"2\.?\s*요약:?([\s\S]*?)(?=3\.|\n\n|$)"
        advice_pattern = r"3\.?\s*조언:?([\s\S]*)"
        
        emotion_match = re.search(emotion_pattern, response, re.IGNORECASE)
        summary_match = re.search(summary_pattern, response, re.IGNORECASE)
        advice_match = re.search(advice_pattern, response, re.IGNORECASE)
        
        emotion_analysis = emotion_match.group(1).strip() if emotion_match else "감정 분석을 찾을 수 없습니다."
        summary = summary_match.group(1).strip() if summary_match else "요약을 찾을 수 없습니다."
        advice = advice_match.group(1).strip() if advice_match else "조언을 찾을 수 없습니다."
        
        # 총점 추출
        total_score_pattern = r"총점:\s*(\d+\.?\d*)/5"
        total_score_match = re.search(total_score_pattern, emotion_analysis)
        total_score = total_score_match.group(1) if total_score_match else "총점을 찾을 수 없습니다."
        
        # 요약에서 줄바꿈 처리
        summary = ' '.join(summary.split())
        
        parsed = {
            "emotion_analysis": emotion_analysis,
            "total_score": total_score,
            "summary": summary,
            "advice": advice
        }
        app.logger.debug(f"Parsed response: {parsed}")
        return parsed
    except Exception as e:
        app.logger.error(f"Error parsing response: {str(e)}\nResponse: {response}")
        return {
            "emotion_analysis": "파싱 오류",
            "total_score": "파싱 오류",
            "summary": "파싱 오류",
            "advice": "파싱 오류"
        }

def analyze_message(assistant_id, user_input):
    try:
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        
        while run.status != 'completed':
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            app.logger.debug(f"Run status: {run.status}")
        
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        answer = messages.data[0].content[0].text.value
        
        app.logger.debug(f"Raw API response: {answer}")
        
        return parse_response(answer)
    except Exception as e:
        app.logger.error(f"Error analyzing message: {str(e)}\n{traceback.format_exc()}")
        raise

@app.route('/get_or_create_assistant', methods=['POST'])
def get_or_create_assistant():
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({"error": "No user_id provided"}), 400
        
        user_id = data['user_id']
        
        if user_id not in user_assistants:
            assistant_id = create_assistant(user_id)
            user_assistants[user_id] = assistant_id
        else:
            assistant_id = user_assistants[user_id]
        
        return jsonify({"assistant_id": assistant_id}), 200
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        app.logger.error(f"Error in get_or_create_assistant: {error_message}\n{traceback.format_exc()}")
        return jsonify({"error": error_message}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'assistant_id' not in data or 'message' not in data:
            return jsonify({"error": "Invalid request data"}), 400
        
        assistant_id = data['assistant_id']
        user_input = data['message']
        
        analysis_result = analyze_message(assistant_id, user_input)
        return Response(json.dumps(analysis_result, ensure_ascii=False), content_type='application/json; charset=utf-8')
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        app.logger.error(f"Error in analyze: {error_message}\n{traceback.format_exc()}")
        return jsonify({"error": error_message}), 500