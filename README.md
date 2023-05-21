# tabasco
tabasco for ReelTaco


## 사용법

1. datasets에 데이터셋 복사 붙여넣기
   * [데이터셋 다운로드(구글 드라이브)](https://drive.google.com/drive/folders/1GeNp7xQDoASuxA2VMnk817FLSbKlomhJ?usp=sharing)
2. .env 파일 생성 후 OPENAI_API_KEY= 'api key' 입력
3. main 파일 돌리기


## output
1. output 형식


   [{
    "index": 0 (number, 각 생성 별로 0~ 붙어있음), 

    "gt_movie": {"movie_title" : string, 

    "rotten_tomatoes_link" : string, "review_score": string(0~1), "review_content": string },
    
    "persona" : string (user의 페르소나 생성 결과, 처음에 None(null)),

    "critic_name" : string (해당 생성에 사용된 유저의 이름),

    "conversation" : string (대화문 생성 결과, 처음에 None(null)),

    "used_ratings" : [ {"review_score", "movie_title", "review_content"}*5],

    "turn_num" : number (대화문 당 얼마나 많은 턴 수를 지정할 것인가),

    "is_casual" : 0 or 1 (캐쥬얼한 대화 생성할 것인지 여부) },
   {}, ...
   ]
2. output/output.json 
   - 해당 파일에 결과 저장
   - 인덱스 순서로 읽어서 conversation이 None 인 것만 api call
   - count 를 세어서 10번째 마다 저장됨(아웃풋 파일에)

3. output/error.json
    - 에러 발생시 해당 파일에 저장 됨 
    - {index, error_message}

