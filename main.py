import os
import csv
import requests
from dotenv import load_dotenv
from typing import List, Dict

# 환경 변수 로드
load_dotenv()

# Perplexity API 설정
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

def fetch_links_direct(query: str) -> List[str]:
    """Perplexity API를 직접 호출하여 링크 가져오기"""
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "당신은 웹사이트 URL을 찾아주는 도우미입니다. 각 질문에 대해 관련 웹사이트 URL만 제공해주세요."},
            {"role": "user", "content": f"'{query}'에 대한 정보를 제공하는 웹사이트 10개를 알려주세요. URL만 나열해주세요."}
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result.get("choices")[0].get("message").get("content", "")
        return extract_links_from_response(content)
    else:
        print(f"API 오류: {response.status_code}")
        print(response.text)
        return []

def extract_links_from_response(response: str) -> List[str]:
    """Perplexity AI 응답에서 링크 추출"""
    import re
    
    # URL 패턴 정규식
    url_pattern = r'https?://\S+'
    
    # 찾은 모든 URL 리스트
    links = re.findall(url_pattern, response)
    
    # 중복 제거 및 상위 10개만 반환
    unique_links = []
    for link in links:
        # 괄호나 마침표 등이 URL의 일부로 포함되었을 경우 제거
        if link.endswith(')') or link.endswith('.') or link.endswith(',') or link.endswith('"'):
            link = link[:-1]
        
        if link not in unique_links:
            unique_links.append(link)
    
    return unique_links[:10]

def save_links_to_csv(links: List[str], filename: str = 'links.csv'):
    """링크를 CSV 파일로 저장"""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 헤더 작성
        writer.writerow(['No.', 'URL'])
        # 데이터 작성
        for idx, link in enumerate(links, 1):
            writer.writerow([idx, link])
    
    print(f"링크가 {filename}에 저장되었습니다.")

def main():
    """메인 프로그램 실행"""
    print("Perplexity AI 검색 링크 수집 프로그램")
    print("------------------------------------")
    
    # 사용자 입력 받기
    query = input("검색할 키워드 또는 내용을 입력하세요: ")
    
    print(f"'{query}'에 대한 검색 중...")
    
    # 링크 가져오기 (직접 API 호출 방식 사용)
    links = fetch_links_direct(query)
    
    if not links:
        print("검색 결과에서 링크를 찾을 수 없습니다.")
        return
    
    # 결과 출력
    print(f"\n검색 결과: {len(links)}개의 링크를 찾았습니다.\n")
    for idx, link in enumerate(links, 1):
        print(f"{idx}. {link}")
    
    # 파일명 설정
    filename = input("\n저장할 CSV 파일명을 입력하세요 (기본값: links.csv): ")
    if not filename:
        filename = "links.csv"
    elif not filename.endswith('.csv'):
        filename += ".csv"
    
    # CSV로 저장
    save_links_to_csv(links, filename)

if __name__ == "__main__":
    main() 