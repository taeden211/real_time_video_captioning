import os
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. 설정 및 모델 로드
# ---------------------------------------------------------
INPUT_FOLDER = "./output"  
COLLECTION_NAME = "construction_safety_v1"

# 한국어 문장 임베딩에 특화된 SBERT 모델
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS" 

print(f" 모델 로드 중: {MODEL_NAME} ...")
embedder = SentenceTransformer(MODEL_NAME)
print(" 모델 로드 완료.")

# ---------------------------------------------------------
# 2. Milvus 연결 및 스키마 정의
# ---------------------------------------------------------
print(" Milvus 연결 시도...")
connections.connect("default", host="localhost", port="19530")

# 기존 컬렉션이 있다면 삭제 
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768), # SBERT 차원수(768)
    FieldSchema(name="template_body", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="hazard_tags", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="source_image", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000) # 원본 설명
]

schema = CollectionSchema(fields, description="건설현장 위험 상황 템플릿 DB")
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f" 컬렉션 생성 완료: {COLLECTION_NAME}")

# ---------------------------------------------------------
# 3. 데이터 로드 및 임베딩
# ---------------------------------------------------------
data_vectors = []
data_templates = []
data_tags = []
data_sources = []
data_descriptions = []

json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
print(f" 총 {len(json_files)}개의 데이터 처리 시작...")

for filename in json_files:
    path = os.path.join(INPUT_FOLDER, filename)
    with open(path, "r", encoding="utf-8") as f:
        item = json.load(f)
        
        # description을 벡터로 변환
        desc = item.get("description", "")
        if not desc: continue
            
        vector = embedder.encode(desc).tolist()
        
        # 태그 리스트를 문자열로 변환 
        tags_str = ",".join(item.get("hazard_tags", []))
        
        data_vectors.append(vector)
        data_templates.append(item.get("template_body", ""))
        data_tags.append(tags_str)
        data_sources.append(item.get("source_image", ""))
        data_descriptions.append(desc)

# ---------------------------------------------------------
# 4. Milvus 데이터 삽입
# ---------------------------------------------------------
if data_vectors:
    entities = [
        data_vectors,
        data_templates,
        data_tags,
        data_sources,
        data_descriptions
    ]
    
    insert_result = collection.insert(entities)
    print(f"데이터 삽입 성공: {insert_result.insert_count}건")
    
    # 검색 인덱스 생성 (속도 향상)
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load() 
    print(" Milvus 로드 완료. 검색 준비 끝.")
else:
    print(" 처리할 데이터가 없습니다.")


test_query = "굴삭기가 덤프트럭에 흙을 싣고잇다."
print(f"Q: '{test_query}'")

query_vec = embedder.encode([test_query])
results = collection.search(
    data=query_vec, 
    anns_field="vector", 
    param={"metric_type": "COSINE", "params": {"nprobe": 10}}, 
    limit=1, 
    output_fields=["template_body", "source_image", "description"]
)

for hits in results:
    for hit in hits:
        print(f"\n Top 1 결과 (유사도: {hit.score:.4f})")
        print(f" - 원본이미지: {hit.entity.get('source_image')}")
        print(f" - 찾은 상황: {hit.entity.get('description')}")
        print(f" - 템플릿: {hit.entity.get('template_body')}")