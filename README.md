KuaiRand 추천 시스템 텀프로젝트
시청 이력만 사용하는 **Item-KNN에 영상 태그를 추가하면 추천 성능이 좋아지는가?**를 확인하는 프로젝트입니다.

인기순 추천, Item-KNN, 하이브리드를 같은 조건에서 비교하고, 기존 사용자 ID를 입력하면 영상 ID와 추천 점수를 최대 10개 출력합니다.

저장소: https://github.com/junhyeokjjangjjangman/kuairand-recsys



빠른 실행 순서
처음 실행할 때

Python 3.13·Git·VS Code 설치 → 저장소 다운로드 → VS Code에서 프로젝트 폴더 열기

→ 가상환경 생성 → 라이브러리 설치 → 원본 CSV 3개를 data/raw에 넣기

→ 아래 파일을 순서대로 실행 → 사용자 ID 5 입력 → 추천 결과 확인

파일 실행 순서

split_data.py → preprocess_data.py → prepare_candidates.py
→ popularity.py → run_item_knn.py → content.py
→ run_hybrid.py → run_test.py → demo.py

한 번 전부 실행한 다음부터

VS Code에서 프로젝트 폴더 열기 → 터미널에서 아래 명령 실행 → 사용자 ID 입력
.\.venv\Scripts\python.exe .\src\demo.py
각 단계의 실제 명령어는 아래 2~6절에 있습니다


1. 실행 환경
아래 안내는 Windows PowerShell 기준입니다.
- 개발 시 사용한 Python: 3.13.7
- 필요한 프로그램: Python 3.13, Git, VS Code
- 주요 라이브러리: pandas, NumPy, SciPy, scikit-learn
- 라이브러리는 requirements.txt로 설치합니다.
GitHub 공동작업자로 초대받은 팀원은 초대를 수락해 주세요.
2. 프로젝트 다운로드
VS Code에서 터미널 → 새 터미널을 열고 PowerShell에서 다음 명령을 실행합니다.
git clone https://github.com/junhyeokjjangjjangman/kuairand-recsys.git
cd kuairand-recsys
VS Code의 파일 → 폴더 열기에서 방금 내려받은 kuairand-recsys 폴더를 열면 코드를 확인할 수 있습니다.

이후 모든 명령은 requirements.txt와 src 폴더가 있는 프로젝트 최상위 폴더에서 실행합니다.

이미 프로젝트를 내려받았다면 해당 폴더에서 다음 명령으로 최신 변경 내용을 받습니다.
git pull origin main
3. Python 환경과 라이브러리 설치
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
아래 실행 명령은 가상환경의 Python 경로를 직접 사용합니다.
4. 원본 데이터 준비
사용하는 데이터는 KuaiRand-Pure입니다. 원본 CSV는 팀에서 별도로 공유받거나 공식 배포 페이지에서 KuaiRand-Pure를 내려받아 준비합니다.

먼저 원본 데이터를 넣을 폴더를 만듭니다.
New-Item -ItemType Directory -Force .\data\raw
아래 세 CSV 파일을 data/raw 바로 안에 넣습니다. 압축을 해제한 폴더가 한 단계 더 들어가지 않도록 최종 경로를 확인합니다.
최종 파일 경로	용도
data/raw/log_standard_4_08_to_4_21_pure.csv	학습용 일반 노출 로그
data/raw/log_random_4_22_to_5_08_pure.csv	검증·테스트용 무작위 노출 로그
data/raw/video_features_basic_pure.csv	영상 ID와 태그 정보


Git에서 제외된 원본 데이터와 data/processed의 생성 파일은 저장소 다운로드만으로 준비되지 않습니다. 처음 실행할 때는 다음 단계에서 전처리 결과와 모델을 생성해야 합니다.
5. 처음 실행할 때: 전처리부터 테스트까지
다음 명령을 위에서부터 하나씩 실행합니다. 현재 명령이 정상적으로 끝나고 터미널 입력줄이 돌아오면 다음 명령을 실행합니다. 오류가 발생하면 해당 단계의 오류를 해결한 뒤 이어서 진행합니다.
.\.venv\Scripts\python.exe .\src\split_data.py
.\.venv\Scripts\python.exe .\src\preprocess_data.py
.\.venv\Scripts\python.exe .\src\prepare_candidates.py
.\.venv\Scripts\python.exe .\src\popularity.py
.\.venv\Scripts\python.exe .\src\run_item_knn.py
.\.venv\Scripts\python.exe .\src\content.py
.\.venv\Scripts\python.exe .\src\run_hybrid.py
.\.venv\Scripts\python.exe .\src\run_test.py
실행 파일	수행 내용
split_data.py	원본 로그를 학습·검증·테스트 기간으로 분리
preprocess_data.py	기간별 중복 사용자–영상 쌍 통합, 학습 관심·노출 이력 저장
prepare_candidates.py	평가 사용자·후보·정답 확정
popularity.py	학습 양성 사용자 수를 이용한 인기순 추천
run_item_knn.py	Item-KNN 모델 생성 및 이웃 수별 검증
content.py	태그 벡터·사용자 취향 벡터 생성 및 콘텐츠 점수 계산
run_hybrid.py	하이브리드 설정별 검증 및 최적 설정 저장
run_test.py	검증에서 선택한 설정을 고정하고 테스트 평가


원본 구조·통계를 확인하거나 평가 함수의 예시 검사를 실행하려면 다음 명령을 추가로 사용할 수 있습니다.
.\.venv\Scripts\python.exe .\src\inspect_data.py
.\.venv\Scripts\python.exe .\src\data_summary.py
.\.venv\Scripts\python.exe .\src\evaluation.py
6. 추천 데모 실행
5단계를 완료한 뒤 실행합니다.
.\.venv\Scripts\python.exe .\src\demo.py
터미널에 User ID:가 나오면 기존 사용자 ID를 입력하고 Enter를 누릅니다. 예를 들어 5를 입력할 수 있습니다.
- 출력: 추천 순위, 영상 ID, 추천 점수 최대 10개
- 저장 예시: results/demo_user_5.csv
- 종료: q 입력 후 Enter
모델 생성과 테스트를 완료한 다음부터는 위 데모 명령으로 실행할 수 있습니다. 데이터나 모델 설정을 변경했다면 관련 전처리·모델 생성·평가 단계를 다시 진행해야 합니다.

데모는 전체 영상 목록에서 해당 사용자의 학습 기간 전체 노출 영상을 제외하고 추천합니다. 양성 학습 이력이 없는 기존 사용자는 인기순 추천을 받습니다.

추천 점수는 후보의 순위를 정하기 위한 값이며, 시청 확률이나 정확도를 뜻하지 않습니다.
7. 현재 실험 조건과 결과
데이터 분할
구분	기간	로그 종류
학습	2022-04-09 ~ 2022-04-21	일반 노출
검증	2022-04-22 ~ 2022-04-28	무작위 노출
테스트	2022-04-29 ~ 2022-05-08	무작위 노출


기간을 먼저 나누고, 각 기간에서 같은 사용자–영상 쌍을 통합합니다. 해당 기간에 한 번이라도 long_view = 1이면 그 쌍의 정답을 1로 처리합니다.

평가 대상은 다음 조건을 만족하는 사용자입니다.
- 학습 기간의 서로 다른 관심 영상이 5개 이상
- 평가 기간에 무작위로 노출된 영상 중 학습 기간 전체 노출 영상을 제외
- 남은 후보가 20개 이상이고, 그 안에 정답 1과 0이 각각 하나 이상 존재
모든 모델은 같은 평가 사용자·후보에서 비교합니다. 검증은 2,322명, 테스트는 7,737명·후보 527,231쌍입니다.
모델 설정
- Item-KNN 이웃 수 탐색: 20, 50, 100
- 하이브리드 CF 비중 탐색: 0.25, 0.50, 0.75
- 선택 기준: 검증 NDCG@10
- 선택된 설정: Item-KNN 이웃 수 100, 하이브리드 이웃 수 100·CF 비중 0.75
- 동점 처리: 점수 내림차순, 영상 ID 오름차순
하이브리드는 사용자별 후보에서 CF와 콘텐츠 점수를 각각 정규화한 뒤 결합합니다. 태그가 없는 후보는 콘텐츠 항을 0으로 두고, 모든 후보에 동일한 CF 가중치를 적용합니다.
Hybrid = alpha × CF_normalized + (1 - alpha) × Content_normalized
수정된 계산 규칙은 설정 파일에 hybrid_scoring_rule: uniform_cf_weight_v2로 기록됩니다. 테스트에는 원래 학습 기간의 이력만 사용하며 검증 이력을 추가하지 않습니다.
수정 후 테스트 결과
모델	평가 사용자 수	Recall@10	NDCG@10
Popularity	7,737	0.363685	0.303846
Item-KNN	7,737	0.346420	0.304554
Hybrid	7,737	0.363614	0.314260


하이브리드는 Item-KNN보다 평균 Recall@10이 약 4.96%, 평균 NDCG@10이 약 3.19% 높았습니다. 인기순 대비 Recall@10 개선은 확인되지 않았습니다.

현재 표는 태그 누락 처리 규칙을 수정한 뒤 같은 테스트 기간에서 재평가한 결과입니다. 원래 테스트를 확인한 뒤 수정한 실험으로 구분해 기록합니다.

평가는 선정된 사용자의 무작위 노출 후보 안에서 수행하며, 데모는 전체 영상 목록을 사용합니다. 이 결과를 전체 사용자나 전체 카탈로그에서의 성능으로 일반화하지 않습니다.
8. 코드와 결과 위치
위치	내용
src/	설정·전처리·모델·평가·데모 코드
src/config.py	데이터 경로·기간·평가 조건·탐색 설정
data/raw/	원본 CSV 세 파일
data/processed/	생성된 전처리 데이터·후보·모델·점수
results/final_test_metrics.csv	세 모델의 테스트 성능표
results/item_knn_validation_metrics.csv	Item-KNN 이웃 수별 검증 결과
results/hybrid_validation_metrics.csv	하이브리드 설정별 검증 결과
results/best_settings.json	검증에서 선택한 설정
results/final_test_settings.json	테스트에 사용한 설정 사본
results/popularity_test_users.csv	인기순 추천의 사용자별 테스트 결과
results/item_knn_test_users.csv	Item-KNN의 사용자별 테스트 결과
results/hybrid_test_users.csv	하이브리드의 사용자별 테스트 결과


9. 실행 중 문제가 생겼을 때
증상	확인할 내용
py -3.13 실행 실패	Python 3.13 설치와 Python 실행 명령을 확인
필요한 라이브러리가 없다는 오류	가상환경의 Python으로 pip install -r requirements.txt 실행
원본 CSV를 찾지 못함	data/raw 바로 안의 파일명과 실행 위치 확인
train_positive.csv, 후보 CSV, .npz 모델 등을 찾지 못함	5단계에서 해당 파일을 만드는 이전 명령이 완료됐는지 확인
Hybrid 계산 규칙이 다르다는 오류	최신 코드로 run_hybrid.py, run_test.py를 순서대로 실행한 뒤 데모 실행
사용자 ID를 찾지 못함	원본 데이터에 있는 기존 사용자 ID를 입력. 예: 5

10. 추후 검토 사항
- 학습·검증·테스트 데이터가 분리되어 있는지
- 관심 이력과 전체 노출 이력이 올바르게 구분되는지
- 모델별 평가 사용자·후보가 동일한지
- Item-KNN·태그 점수·하이브리드 결합 계산이 정확한지
- 평가 지표 계산과 동점 처리가 일관적인지
- 평가 대상의 대표성, 설정 탐색 범위, 결과 해석에 추가 검토가 필요한지