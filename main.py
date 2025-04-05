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
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

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
        retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
        chain_type="stuff",
        return_source_documents=True
    )
    
    # 단일 문자열 프롬프트 사용으로 변경 (기존 방식으로 돌아가기)
    system_content = """
당신은 SEO에 능숙한 전문 블로그 작가입니다. 다음 지침을 준수하여 SEO 친화적이고 풍부한 콘텐츠의 블로그를 작성하십시오.

주요 키워드를 자연스럽게 제목과 소제목(H2, H3)에 삽입하세요.

본문에는 키워드를 적절한 빈도로 반복하면서, 키워드 동의어나 관련 검색어를 함께 사용하여 내용의 밀도를 높이세요.

각 섹션은 명확한 주제를 다루고, 독자가 실제로 얻을 수 있는 구체적이고 실질적인 정보를 포함하세요.


블로그는 서론, 본론, 결론 구조를 갖추고, 제목과 소제목을 한 번만 사용하여 중복 없이 작성해야 합니다.
"""
    
    assistant_content = "네, 기존 블로그 내용을 꼼꼼히 분석한 후 SEO 친화적인 제목과 내용을 추가하여 블로그를 풍부하게 완성하겠습니다."
    
    human_content = f"아래 블로그 내용에 키워드에 맞추어 SEO 친화적인 새로운 섹션을 추가하고, 글의 마무리를 결론 섹션으로 작성해주세요. 주제: {query}"
    
    # 전체 프롬프트 생성
    prompt = f"""
[시스템]
{system_content}

[어시스턴트]
{assistant_content}

[휴먼]
{human_content}
"""
    
    try:
        # 기존 방식으로 호출 (원래 작동하던 방식)
        result = qa_chain.invoke({"query": prompt})
        generated_text = result["result"]
        
        # 결과가 너무 짧은 경우 보완 요청
        if len(generated_text) < 2000:
            # 원본 텍스트 분석
            sections = extract_sections(generated_text)
            
            # 보완 요청 프롬프트 작성
            supplement_content_system = """
            당신은 SEO에 능숙한 전문 블로그 작가입니다. 기존 블로그 내용을 분석하고 부족한 부분을 보완하는 것이 임무입니다.
            
            중요: 기존 콘텐츠의 구조와 내용을 유지하되, 추가 정보와 설명으로 내용을 풍부하게 만들어야 합니다.
            중요: 절대로 기존 섹션을 반복하지 마세요. 기존 섹션을 유지하고 새로운 정보를 추가하는 방식으로 작성하세요.
            
            다음 지침을 준수하세요:
            1. 기존 제목과 구조를 유지하세요. 같은 제목을 반복하지 마세요.
            2. 기존 내용이 충분하지 않은 섹션에 상세 정보를 추가하세요.
            3. 필요한 경우에만 새로운 소제목(H3, H4)을 추가하세요.
            4. 전체적인 글의 흐름과 논리가 자연스럽게 이어지도록 작성하세요.
            5. 결론 섹션이 없는 경우 결론을 추가하세요.
            6. 키워드를 자연스럽게 반복하여 SEO 최적화하세요.

            """
            
            supplement_content_assistant = "네, 기존 글의 구조와 내용을 유지하면서 부족한 부분을 보완하겠습니다."
            
            supplement_content_human = f"""
            아래 블로그를 분석하고, 부족한 부분을 보완하여 완성된 블로그 글을 작성해주세요.
            키워드 '{query}'에 맞게 SEO 친화적인 내용으로 보완해주세요.
            
            기존 블로그 내용:
            {generated_text}
            """
            
            supplement_prompt = f"""
            [시스템]
            {supplement_content_system}
            
            [어시스턴트]
            {supplement_content_assistant}
            
            [휴먼]
            {supplement_content_human}
            """
            
            supplement_result = qa_chain.invoke({"query": supplement_prompt})
            supplement_text = supplement_result["result"]
            
            # 기존 글과 보완 글을 섹션 단위로 통합
            integrated_text = integrate_blog_contents(generated_text, supplement_text)
            
            # 중복 섹션 제거 및 순서 조정
            cleaned_text = remove_duplicate_sections(integrated_text)
            
            # 최종 확인과 정리
            final_content_system = """
            당신은 전문 블로그 편집자입니다. 다음 글에서 구조적 문제가 있는지 확인하고 최종 정리를 진행하세요.
            
            점검 사항:
            1. 서론, 본론, 결론 순서가 올바른지 확인
            2. 소제목이 본론 앞에 오지 않도록 조정
            3. 제목 중복이 없는지 확인
            4. 문장 끝에 불필요한 마침표가 중복되지 않는지 확인
            5. 불필요하게 반복되는 문단이 있는지 확인

            7. SEO 관점에서 키워드가 제목, 소제목, 본문에 적절히 분포되어 있는지 확인
            """
            
            final_content_assistant = "네, 제공된 글의 구조적 문제를 확인하고 최종 정리하겠습니다."
            
            final_content_human = f"""
            다음 블로그 글의 구조와 형식을 최종 점검하고, 필요한 수정을 진행한 후 최종 정리된 글을 제공해주세요.
            주제 키워드는 '{query}'입니다. 글 마지막에 160자 이내의 메타 디스크립션을 반드시 추가해주세요.
            
            블로그 내용:
            {cleaned_text}
            """
            
            final_prompt = f"""
            [시스템]
            {final_content_system}
            
            [어시스턴트]
            {final_content_assistant}
            
            [휴먼]
            {final_content_human}
            """
            
            final_result = qa_chain.invoke({"query": final_prompt})
            final_text = final_result["result"]
            
            return final_text
        
        # 원본 결과가 충분히 긴 경우, 중복 제거 및 구조 정리만 수행
        cleaned_text = remove_duplicate_sections(generated_text)
        
        # 최종 확인과 정리 (메타 디스크립션 추가 등)
        final_content_system = """
        당신은 전문 블로그 편집자입니다. 다음 글에서 구조적 문제가 있는지 확인하고 최종 정리를 진행하세요.
        
        점검 사항:
        1. 서론, 본론, 결론 순서가 올바른지 확인
        2. 소제목이 본론 앞에 오지 않도록 조정
        3. 제목 중복이 없는지 확인
        4. 문장 끝에 불필요한 마침표가 중복되지 않는지 확인
        5. 불필요하게 반복되는 문단이 있는지 확인
        6. 메타 디스크립션이 글 맨 마지막에 있는지 확인 (없다면 추가)
        7. SEO 관점에서 키워드가 제목, 소제목, 본문에 적절히 분포되어 있는지 확인
        """
        
        final_content_assistant = "네, 제공된 글의 구조적 문제를 확인하고 최종 정리하겠습니다."
        
        final_content_human = f"""
        다음 블로그 글의 구조와 형식을 최종 점검하고, 필요한 수정을 진행한 후 최종 정리된 글을 제공해주세요.
        주제 키워드는 '{query}'입니다. 글 마지막에 160자 이내의 메타 디스크립션을 반드시 추가해주세요.
        
        블로그 내용:
        {cleaned_text}
        """
        
        final_prompt = f"""
        [시스템]
        {final_content_system}
        
        [어시스턴트]
        {final_content_assistant}
        
        [휴먼]
        {final_content_human}
        """
        
        final_result = qa_chain.invoke({"query": final_prompt})
        final_text = final_result["result"]
        
        return final_text
        
    except Exception as e:
        print(f"글 생성 중 오류 발생: {e}")
        print(f"오류 상세: {str(e)}")
        return "블로그 글 생성에 실패했습니다. 다시 시도해주세요."

def extract_sections(text: str) -> dict:
    """블로그 글에서 섹션별로 내용 추출"""
    lines = text.split('\n')
    sections = {}
    current_section = None
    current_content = []
    
    for line in lines:
        # 제목 확인 (H1, H2, H3 수준)
        is_header = False
        header_level = 0
        
        for level, marker in enumerate(['# ', '## ', '### '], 1):
            if line.startswith(marker):
                is_header = True
                header_level = level
                header_text = line.strip()
                
                # 이전 섹션 저장
                if current_section and current_content:
                    sections[current_section] = {
                        'level': current_section_level,
                        'content': '\n'.join(current_content)
                    }
                
                # 새 섹션 시작
                current_section = header_text
                current_section_level = header_level
                current_content = []
                break
        
        if not is_header and current_section is not None:
            current_content.append(line)
    
    # 마지막 섹션 저장
    if current_section and current_content:
        sections[current_section] = {
            'level': current_section_level,
            'content': '\n'.join(current_content)
        }
    
    return sections

def integrate_blog_contents(original_text: str, supplement_text: str) -> str:
    """초기 블로그 글과 보완 글을 통합"""
    # 각 텍스트에서 섹션 추출
    original_sections = extract_sections(original_text)
    supplement_sections = extract_sections(supplement_text)
    
    # 결과 저장을 위한 변수
    integrated_lines = []
    
    # 섹션별 우선순위 설정 (일반적인 블로그 구조)
    priority_sections = {
        "# ": 1,        # 메인 제목 (H1)
        "## 서론": 2,    # 서론 (H2)
        "### 서론": 2,
        "## 본론": 3,    # 본론 (H2)
        "### 본론": 3,
        "## 결론": 99,   # 결론 (H2)
        "### 결론": 99,
    }
    
    # 통합된 섹션 목록 생성
    all_sections = {}
    
    # 먼저 원본 섹션 추가
    for header, section_info in original_sections.items():
        all_sections[header] = {
            'level': section_info['level'],
            'content': section_info['content'],
            'priority': get_section_priority(header, priority_sections)
        }
    
    # 보완 섹션 추가 또는 통합
    for header, section_info in supplement_sections.items():
        # 이미 존재하는 섹션인지 확인
        existing_header = find_similar_header(header, all_sections.keys())
        
        if existing_header:
            # 동일한 섹션이 있는 경우, 내용 통합 (중복 제거 및 원본 우선)
            original_content = all_sections[existing_header]['content']
            supplement_content = section_info['content']
            
            # 내용이 추가된 경우에만 통합
            if len(supplement_content) > len(original_content) * 1.2:  # 20% 이상 내용 증가 시 통합
                combined_content = combine_contents(original_content, supplement_content)
                all_sections[existing_header]['content'] = combined_content
        else:
            # 새로운 섹션인 경우, 추가
            priority = get_section_priority(header, priority_sections)
            all_sections[header] = {
                'level': section_info['level'],
                'content': section_info['content'],
                'priority': priority
            }
    
    # 섹션 우선순위에 따라 정렬
    sorted_sections = sorted(all_sections.items(), key=lambda x: (x[1]['priority'], x[1]['level']))
    
    # 모든 섹션 통합
    for header, section_info in sorted_sections:
        integrated_lines.append(header)
        if header.strip() == "## 본론":
            # 본론 다음에 H3 제목들이 오도록 변경
            h3_sections = [sect for sect in sorted_sections if sect[0].startswith('### ') and sect[1]['priority'] > 2 and sect[1]['priority'] < 99]
            if h3_sections:
                for h3_header, h3_info in h3_sections:
                    integrated_lines.append('')
                    integrated_lines.append(h3_header)
                    content_lines = h3_info['content'].split('\n')
                    integrated_lines.extend(content_lines)
                    integrated_lines.append('')
                # 본론 내용은 H3 뒤에 추가하지 않음
                continue
        
        content_lines = section_info['content'].split('\n')
        integrated_lines.extend(content_lines)
        integrated_lines.append('')  # 섹션 사이 빈 줄 추가
    
    # 결과 텍스트 생성
    return '\n'.join(integrated_lines)

def get_section_priority(header: str, priority_map: dict) -> int:
    """섹션 헤더의 우선순위 반환"""
    for key, priority in priority_map.items():
        if key in header.lower():
            return priority
    
    # H1 제목이면 최상위 우선순위
    if header.startswith('# '):
        return 1
    # H2 섹션이면 중간 우선순위
    elif header.startswith('## '):
        return 50
    # H3 이하 섹션이면 낮은 우선순위
    else:
        return 75

def find_similar_header(header: str, existing_headers: list) -> str:
    """유사한 헤더 찾기"""
    # 정확히 일치하는 경우
    if header in existing_headers:
        return header
    
    # 내용이 유사한 경우
    header_text = header.split(' ', 1)[1] if ' ' in header else header
    for existing in existing_headers:
        existing_text = existing.split(' ', 1)[1] if ' ' in existing else existing
        # 헤더 텍스트가 80% 이상 일치하는 경우
        if header_text in existing_text or existing_text in header_text:
            return existing
    
    return None

def combine_contents(original: str, supplement: str) -> str:
    """두 내용을 중복 없이 통합"""
    # 원본 내용을 문장 단위로 분리
    original_sentences = [s.strip() for s in original.replace('\n', ' ').split('.') if s.strip()]
    supplement_sentences = [s.strip() for s in supplement.replace('\n', ' ').split('.') if s.strip()]
    
    # 중복 문장 제거
    unique_sentences = []
    for sentence in original_sentences:
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    for sentence in supplement_sentences:
        is_duplicate = False
        for orig in original_sentences:
            # 문장이 80% 이상 일치하는 경우 중복으로 간주
            similarity = len(set(sentence.split()) & set(orig.split())) / max(len(set(sentence.split())), len(set(orig.split())) or 1)
            if similarity > 0.8:
                is_duplicate = True
                break
        
        if not is_duplicate and sentence:
            unique_sentences.append(sentence)
    
    # 순서대로 내용 결합 (보완 내용은 원본 내용 뒤에 추가)
    combined_text = '. '.join(unique_sentences)
    if not combined_text.endswith('.'):
        combined_text += '.'
    
    # 문단으로 재구성
    paragraphs = []
    current_para = []
    
    for sentence in combined_text.split('. '):
        current_para.append(sentence)
        if len(current_para) >= 5:  # 5문장마다 문단 구분
            paragraphs.append('. '.join(current_para) + '.')
            current_para = []
    
    if current_para:
        paragraphs.append('. '.join(current_para) + '.')
    
    return '\n\n'.join(paragraphs)

def remove_duplicate_sections(text: str) -> str:
    """블로그 글에서 중복된 섹션 제거"""
    lines = text.split('\n')
    
    # 제목과 소제목을 추적하기 위한 세트
    seen_headers = set()
    filtered_lines = []
    
    # 중복 서론 있는지 확인 및 제거
    intro_sections = ['## 서론', '### 서론']
    seen_intro = False
    
    # 중복 본론 있는지 확인 및 제거
    content_sections = ['## 본론', '### 본론']
    seen_content = False
    
    # 중복 결론 있는지 확인 및 제거
    conclusion_sections = ['## 결론', '### 결론']
    seen_conclusion = False

    # H3 헤더가 본론 앞에 나타날 경우를 체크
    main_content_index = -1
    h3_headers_before_main = []
    h3_headers_with_content = []
    
    # 먼저 구조 파악 (본론 위치와 H3 헤더 위치)
    for i, line in enumerate(lines):
        if any(line.startswith(section) for section in content_sections):
            main_content_index = i
            break
    
    # 본론 섹션이 있으면 구조 재배치 검토
    if main_content_index > 0:
        i = 0
        while i < len(lines):
            line = lines[i]
            # H3 헤더 찾기
            if line.startswith('### ') and i < main_content_index:
                header = line
                content = []
                i += 1
                # 헤더 다음 내용 수집
                while i < len(lines) and not (lines[i].startswith('# ') or lines[i].startswith('## ') or lines[i].startswith('### ')):
                    content.append(lines[i])
                    i += 1
                h3_headers_before_main.append((header, content))
                continue
            i += 1
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 빈 줄은 유지
        if not line.strip():
            filtered_lines.append(line)
            i += 1
            continue
            
        # 제목 및 소제목 확인
        is_header = False
        for marker in ['# ', '## ', '### ', '#### ', '##### ', '###### ']:
            if line.startswith(marker):
                is_header = True
                header_text = line.strip()
                
                # 서론 관련 제목 확인
                if any(intro in header_text.lower() for intro in ['서론', '소개', 'introduction']):
                    if seen_intro:
                        # 이미 서론 섹션이 있으면 스킵
                        is_header = 'skip'
                    else:
                        seen_intro = True
                
                # 본론 관련 제목 확인
                elif any(content in header_text.lower() for content in ['본론', '내용', 'content']):
                    if seen_content:
                        # 이미 본론 섹션이 있으면 스킵
                        is_header = 'skip'
                    else:
                        seen_content = True
                        # 본론을 추가할 때 앞서 찾았던 H3 헤더들을 추가
                        filtered_lines.append(header_text)
                        filtered_lines.append("")  # 빈 줄 추가
                        
                        # 본론 앞에 있던 H3 헤더와 내용을 본론 아래로 이동
                        for h3_header, h3_content in h3_headers_before_main:
                            filtered_lines.append(h3_header)
                            filtered_lines.extend(h3_content)
                            filtered_lines.append("")  # 빈 줄 추가
                        
                        is_header = 'processed'  # 이미 처리했다고 표시
                
                # 결론 관련 제목 확인
                elif any(conclusion in header_text.lower() for conclusion in ['결론', '마무리', 'conclusion']):
                    if seen_conclusion:
                        # 이미 결론 섹션이 있으면 스킵
                        is_header = 'skip'
                    else:
                        seen_conclusion = True
                
                # 다른 제목들은 중복 체크
                elif header_text in seen_headers:
                    is_header = 'skip'
                else:
                    seen_headers.add(header_text)
                
                break
        
        # 헤더가 중복되었으면 해당 섹션 전체를 건너뛰기 위한 플래그
        if is_header == 'skip':
            i += 1
            # 다음 헤더가 나올 때까지 스킵
            while i < len(lines) and not any(lines[i].startswith(marker) for marker in ['# ', '## ', '### ', '#### ', '##### ', '###### ']):
                i += 1
            continue
        elif is_header == 'processed':
            # 이미 처리한 경우 다음 줄로
            i += 1
            continue
            
        # 중복되지 않은 라인 추가
        filtered_lines.append(line)
        i += 1
    
    # 결론이 없으면 기본 결론 추가
    if not seen_conclusion:
        filtered_lines.append("\n## 결론")
        filtered_lines.append("\n벚꽃은 봄의 상징으로, 아름다운 경관과 함께 사람들에게 행복한 추억을 선사합니다. 위에서 소개한 장소들은 각기 다른 매력을 지니고 있어, 방문객들에게 특별한 경험을 제공합니다. 벚꽃 시즌에는 이들 명소를 찾아가 꽃의 아름다움을 만끽해 보시길 바랍니다. 꽃잎이 흩날리는 순간, 그 아름다움은 여러분의 마음에 오래도록 남을 것입니다.")
    
    return '\n'.join(filtered_lines)

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
            {"role": "user", "content": f"'{query}'에 대한 정보를 제공하는 웹사이트 20개를 알려주세요. URL만 나열해주세요."}
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
    
    return unique_links[:20]

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