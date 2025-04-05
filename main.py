import os
import csv
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from playwright.sync_api import sync_playwright
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import redis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import json
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# API 키 설정
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Redis 클라이언트 초기화
redis_client = redis.from_url(REDIS_URL)

def save_to_cache(key: str, data):
    """데이터를 Redis 캐시에 저장"""
    try:
        # 문자열이든 딕셔너리든 모두 처리할 수 있도록 수정
        if isinstance(data, dict):
            redis_client.setex(key, 3600, json.dumps(data))
        else:
            redis_client.setex(key, 3600, data)
    except Exception as e:
        print(f"캐시 저장 중 오류 발생: {e}")

def get_from_cache(key: str):
    """Redis 캐시에서 데이터 조회"""
    try:
        data = redis_client.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data.decode('utf-8')
        return None
    except Exception as e:
        print(f"캐시 조회 중 오류 발생: {e}")
        return None

def create_vector_store(texts: List[str]):
    """텍스트를 임베딩하여 벡터 저장소 생성"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_text("\n".join(texts))
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(splits, embeddings)
    return vectorstore

def generate_blog_post(query: str, vectorstore) -> str:
    """RAG를 사용하여 블로그 글 생성"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        chain_type="stuff",
        return_source_documents=True
    )
    
    prompt = f"""
    다음 주제에 대해 제공된 모든 정보를 활용하여 상세한 블로그 글을 작성해주세요.
    주제: {query}
    """
    
    try:
        # deprecation 경고 해결: __call__ 대신 invoke 사용
        result = qa_chain.invoke({"query": prompt})
        generated_text = result["result"]
        
        # 결과가 너무 짧은 경우 보완 요청
        if len(generated_text) < 2000:
            supplement_prompt = f"""
            앞서 생성된 글의 내용이 충분하지 않습니다. 내용을 추가로 보완해주세요.
            
            기존 내용:
            {generated_text}
            """
            
            supplement_result = qa_chain.invoke({"query": supplement_prompt})
            generated_text = supplement_result["result"]
        
        return generated_text
        
    except Exception as e:
        print(f"글 생성 중 오류 발생: {e}")
        print(f"오류 상세: {str(e)}")
        return "블로그 글 생성에 실패했습니다. 다시 시도해주세요."

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
    
    url_pattern = r'https?://\S+'
    links = re.findall(url_pattern, response)
    
    unique_links = []
    for link in links:
        if link.endswith(')') or link.endswith('.') or link.endswith(',') or link.endswith('"'):
            link = link[:-1]
        
        if link not in unique_links:
            unique_links.append(link)
    
    return unique_links[:10]

def crawl_with_playwright(url: str) -> Tuple[bool, str]:
    """Playwright를 사용하여 웹페이지 크롤링"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, timeout=30000)
            
            # 페이지 로딩 대기
            page.wait_for_load_state('networkidle')
            
            # 스크롤을 통해 동적 콘텐츠 로드
            page.evaluate("""() => {
                window.scrollTo(0, document.body.scrollHeight);
                return new Promise(resolve => setTimeout(resolve, 2000));
            }""")
            
            # 불필요한 요소만 제거 (스크립트와 스타일만)
            page.evaluate("""() => {
                const elements = document.querySelectorAll('script, style');
                elements.forEach(el => el.remove());
            }""")
            
            # 모든 텍스트 내용 추출
            text = page.evaluate("""() => {
                // 모든 텍스트 노드의 내용을 수집
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );

                let textContent = [];
                let node;
                while (node = walker.nextNode()) {
                    const text = node.textContent.trim();
                    if (text) {
                        textContent.push(text);
                    }
                }
                
                return textContent.join('\\n');
            }""")
            
            browser.close()
            return True, text
    except Exception as e:
        return False, str(e)

def crawl_with_selenium(url: str) -> Tuple[bool, str]:
    """Selenium을 사용하여 웹페이지 크롤링"""
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # 페이지 로딩 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # 스크롤을 통해 동적 콘텐츠 로드
        driver.execute_script("""
            window.scrollTo(0, document.body.scrollHeight);
        """)
        time.sleep(2)  # 동적 콘텐츠 로딩 대기
        
        # 불필요한 요소만 제거 (스크립트와 스타일만)
        driver.execute_script("""
            const elements = document.querySelectorAll('script, style');
            elements.forEach(el => el.remove());
        """)
        
        # 모든 텍스트 내용 추출
        text = driver.execute_script("""
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            let textContent = [];
            let node;
            while (node = walker.nextNode()) {
                const text = node.textContent.trim();
                if (text) {
                    textContent.push(text);
                }
            }
            
            return textContent.join('\\n');
        """)
        
        driver.quit()
        return True, text
    except Exception as e:
        return False, str(e)

def crawl_links(links: List[str]) -> List[Dict]:
    """모든 링크를 크롤링하고 결과를 반환"""
    results = []
    
    for url in links:
        # 캐시 확인
        cached_data = get_from_cache(f"crawl:{url}")
        if cached_data:
            print(f"캐시에서 {url} 데이터를 찾았습니다.")
            results.append(cached_data)
            continue
            
        print(f"\n{url} 크롤링 중...")
        
        # Playwright로 먼저 시도
        success, content = crawl_with_playwright(url)
        
        # 실패하면 Selenium으로 재시도
        if not success:
            print(f"Playwright 크롤링 실패, Selenium으로 재시도 중...")
            success, content = crawl_with_selenium(url)
        
        if success:
            print(f"✅ 성공: {url}")
            result = {
                'url': url,
                'content': content,
                'status': 'success'
            }
            results.append(result)
            # 캐시에 저장
            save_to_cache(f"crawl:{url}", result)
        else:
            print(f"❌ 실패: {url}")
            print(f"실패 원인: {content}")
            results.append({
                'url': url,
                'content': f"크롤링 실패: {content}",
                'status': 'failed'
            })
        
        # 서버 부하 방지를 위한 대기
        time.sleep(random.uniform(1, 3))
    
    return results

def save_results_to_csv(results: List[Dict], filename: str):
    """크롤링 결과를 CSV 파일로 저장"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n결과가 {filename}에 저장되었습니다.")

def main():
    """메인 프로그램 실행"""
    print("블로그 글 생성 프로그램")
    print("--------------------------------------------")
    
    # 사용자 입력 받기
    query = input("블로그 글의 주제를 입력하세요: ")
    
    print(f"'{query}'에 대한 검색 중...")
    
    # 링크 가져오기
    links = fetch_links_direct(query)
    
    if not links:
        print("검색 결과에서 링크를 찾을 수 없습니다.")
        return
    
    # 크롤링 실행
    print("\n크롤링을 시작합니다...")
    results = crawl_links(links)
    
    # 성공적으로 크롤링된 내용만 추출
    successful_contents = [r['content'] for r in results if r['status'] == 'success']
    
    if not successful_contents:
        print("크롤링에 실패했습니다.")
        return
    
    # 벡터 저장소 생성
    print("\n벡터 저장소 생성 중...")
    vectorstore = create_vector_store(successful_contents)
    
    # 블로그 글 생성
    print("\n블로그 글 생성 중...")
    blog_post = generate_blog_post(query, vectorstore)
    
    # 결과 출력
    print("\n생성된 블로그 글:")
    print(blog_post)
    
    # 현재 시간 가져오기
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 파일로 저장 (타임스탬프 추가)
    filename = f"blog_{query.replace(' ', '_')}_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(blog_post)
    print(f"\n블로그 글이 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    main() 