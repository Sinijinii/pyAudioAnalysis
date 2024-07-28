from konlpy.tag import Okt
from difflib import get_close_matches

# 미리 정의된 색상, 패턴, 종류 및 브랜드
colors = {
    '빨간색': (255, 0, 0),
    '파란색': (0, 0, 255),
    '노란색': (255, 255, 0),
    '초록색': (0, 255, 0),
    '검은색': (0, 0, 0),
    '흰색': (255, 255, 255),
    '핑크': (255, 192, 203)
}
types = ['니트', '셔츠', '반팔', '긴팔', '코트', '재킷']
patterns = ['스트라이프', '체크', '도트', '플라워']
brands = []

# RGB 값을 사용하여 가장 가까운 색상을 찾는 함수
def find_closest_color(color_name):
    if color_name in colors:
        return color_name
    
    # 코랄색 등 입력된 색상에 가장 가까운 색상 찾기
    color_names = list(colors.keys())
    closest_color = get_close_matches(color_name, color_names, n=1, cutoff=0.1)
    
    if closest_color:
        return closest_color[0]
    else:
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
        closest_color = find_closest_color(word)
        if closest_color:
            extracted_colors.append(closest_color)
        if word in types:
            extracted_types.append(word)
        if word in patterns:
            extracted_patterns.append(word)
        if word in brands:
            extracted_brands.append(word)
    
    return extracted_colors, extracted_types, extracted_patterns, extracted_brands

# 사용자 입력 예제
sentence = "나는 오늘 코랄색 스트라이프 니트를 입을거야"
colors, types, patterns, brands = extract_clothing_features(sentence)

print(f"추출된 색상: {', '.join(colors)}")
print(f"추출된 종류: {', '.join(types)}")
print(f"추출된 패턴: {', '.join(patterns)}")
print(f"추출된 브랜드: {', '.join(brands)}")
