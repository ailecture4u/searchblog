# Perplexity AI 검색 링크 수집 프로그램

Perplexity AI를 이용하여 특정 키워드에 대한 검색 결과 링크를 수집하고 CSV 파일로 저장하는 프로그램입니다.

## 기능

- Perplexity AI를 통한 키워드 검색
- 검색 결과에서 상위 10개 링크 추출
- 추출된 링크를 CSV 파일로 저장

## 설치 방법

1. 레포지토리 복제
```bash
git clone https://github.com/username/perplexity-search-links.git
cd perplexity-search-links
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
   - `.env.example` 파일을 `.env`로 복사하고 Perplexity API 키를 입력하세요.
```bash
cp .env.example .env
```
   - `.env` 파일을 편집하여 `PERPLEXITY_API_KEY` 값을 설정하세요.

## 사용 방법

1. 프로그램 실행
```bash
python main.py
```

2. 검색할 키워드 또는 내용 입력
   - 프로그램이 실행되면 검색할 키워드를 입력하라는 메시지가 표시됩니다.
   - 키워드를 입력하면 Perplexity AI가 검색을 시작합니다.

3. 검색 결과 확인 및 CSV 저장
   - 검색 결과로 찾은 링크가 화면에 표시됩니다.
   - 저장할 CSV 파일명을 입력하면 해당 이름으로 파일이 저장됩니다.

## CSV 파일 형식

생성된 CSV 파일은 다음과 같은 형식을 가집니다:

| No. | URL                              |
|-----|----------------------------------|
| 1   | https://www.example.com/result1  |
| 2   | https://www.example.com/result2  |
| …   | …                                |
| 10  | https://www.example.com/result10 |

## 주의 사항

- Perplexity API는 사용량 제한이 있을 수 있으므로 API 사용량을 확인하세요.
- API 키는 절대 공개 레포지토리에 커밋하지 마세요. 