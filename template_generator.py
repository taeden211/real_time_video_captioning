import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# -------------------------------------------------
# 1. 환경 변수 및 설정
# -------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

client = OpenAI(api_key=api_key)

# -------------------------------------------------
# 2. 이미지 처리 함수
# -------------------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_template_from_image(image_path):
    base64_image = encode_image(image_path)
    
    system_prompt = """
    You are a Vision-Language data generator for construction safety.
    Analyze the image and output a JSON object only. 
    Do not include markdown formatting (```json) or conversational text.
    """
    
    user_prompt = """
    Analyze the provided construction site image and generate a JSON object with the following fields:

    1. "description": 
       - Factual, dry description for vector indexing. 
       - Format: [Subject] [Action] at [Location] with [Condition].
       - Language: Korean.
       - Example: "근로자가 안전모를 착용하지 않고 2층 비계 위를 걷고 있다."

    2. "template_body": 
       - Warning caption using Jinja2 syntax.
       - Allowed variables: {{ object }}, {{ action }}, {{ location }}, {{ hazard }}.
       - Style: Concise, warning tone.
       - Example: "[경고] {{ location }} 내 {{ object }} {{ hazard }} 위험 감지."

    3. "variables": 
       - Actual values extracted from the image corresponding to the template variables.

    4. "hazard_tags": 
       - List of 3-5 keywords for filtering (e.g., ["추락", "비계", "안전대"]).

    [Strict Vocabulary Rules]
    - Person -> "근로자"
    - Scaffold -> "비계"
    - Support prop -> "동바리"
    - Forklift -> "지게차"
    - Helmet -> "안전모"
    - Safety Belt -> "안전대"
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# -------------------------------------------------
# 3. 메인 실행 (개별 파일 저장 로직)
# -------------------------------------------------
if __name__ == "__main__":
    input_folder = "./dataset"      # 이미지가 있는 폴더
    output_folder = "./output" # 결과가 저장될 폴더

    # 결과 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Error: '{input_folder}' folder not found.")
        exit()

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(files)

    print(f"Starting processing for {total} images...")

    for idx, filename in enumerate(files):
        img_path = os.path.join(input_folder, filename)
        
        # 출력 파일명 생성 (예: image.jpg -> image.json)
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}.json"
        output_path = os.path.join(output_folder, json_filename)

        print(f"[{idx+1}/{total}] Processing: {filename}...", end=" ")

        try:
            # 이미 처리된 파일이 있다면 건너뛰기
            if os.path.exists(output_path):
                print("Skipped (Already exists).")
                continue

            template = generate_template_from_image(img_path)
            
            # 소스 이미지 정보 추가
            template['source_image'] = filename 
            
            #  개별 JSON 파일로 즉시 저장
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
            
            print(f"Saved to {json_filename}.")

        except Exception as e:
            print(f"\n -> Error processing {filename}: {e}")
            # 에러가 나도 다음 파일로 계속 진행됨

    print(f"\n All tasks completed. Check the '{output_folder}' folder.")