from konlpy.tag import Okt
from difflib import get_close_matches
from brands import brand_synonyms
from colors import color_synonyms,colors

types = {
    "니트": ["니트", "스웨터"],
    "셔츠": ["셔츠", "와이셔츠", "드레스 셔츠"],
    "반팔": ["반팔", "반소매", "티셔츠"],
    "긴팔": ["긴팔", "긴소매", "롱슬리브"],
    "코트": ["코트", "오버코트", "트렌치코트"],
    "재킷": ["재킷", "자켓", "블레이저"]
}

patterns = {
    "단색": ["단색", "무지"],
    "드로잉": ["드로잉", "그림"],
    "로고/그래픽": ["로고", "그래픽"],
    "타이다이": ["타이다이"],
    "그라데이션": ["그라데이션"],
    "컬러블록": ["컬러블록", "색상블록"],
    "스트라이프": ["스트라이프", "줄무늬"],
    "도트": ["도트", "물방울","땡땡이"],
    "체크": ["체크", "격자"],
    "플라워": ["플라워", "꽃무늬"]
}


# 유사어 사전을 사용하여 색상 및 브랜드 추출
def find_closest_color(word):
    for color, synonyms in color_synonyms.items():
        if word == color or word in synonyms:
            return colors[color]
    return None

def find_closest_brand(word):
    for brand, synonyms in brand_synonyms.items():
        if word == brand or word in synonyms:
            return brand
    return None

# 문장에서 색상, 패턴, 종류, 브랜드를 추출하는 함수
def extract_clothing_features(sentence):
    okt = Okt()
    tokens = okt.pos(sentence, norm=True, stem=True)
    
    extracted_colors = []
    extracted_types = []
    extracted_patterns = []
    extracted_brands = []
    
    for word, pos in tokens:
        if pos == 'Noun':  # 명사만 고려
            closest_color = find_closest_color(word)
            if closest_color:
                extracted_colors.append(closest_color)
            closest_brand = find_closest_brand(word)
            if closest_brand:
                extracted_brands.append(closest_brand)
            if word in types:
                extracted_types.append(word)
            if word in patterns:
                extracted_patterns.append(word)
    
    return extracted_colors, extracted_types, extracted_patterns, extracted_brands

# 사용자 입력 예제
sentence = "나 오늘 하얀색 폴로 니트 입고싶어."
extracted_colors, extracted_types, extracted_patterns, extracted_brands = extract_clothing_features(sentence)

print(f"추출된 색상: {', '.join(extracted_colors)}")
print(f"추출된 종류: {', '.join(extracted_types)}")
print(f"추출된 패턴: {', '.join(extracted_patterns)}")
print(f"추출된 브랜드: {', '.join(extracted_brands)}")