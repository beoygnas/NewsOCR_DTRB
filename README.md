# 딥러닝 프로젝트

## idea

idea : 신문 이미지 → text → summarization 을 하는 앱

1차 목표 : 신문기사 이미지를, text로 반환 

CRAFT : 기존의 한 단어, 단위로 하던 바운딩박스를 한 문단으로 묶어주기

DTRB : 기존의 영어 모델이기 때문에, 한국어로 파인튜닝

파인튜닝 시에, 일반 한국어 데이터 vs 기사 데이터 를 비교하면 좋을듯

2차 목표 : text화 된 한국어 기사를  한 문장으로 요약

KoBART-summarization 을 AI  hub로 fine tuning 

3차 목표 : 갤러리 접근 / 요약기능을 하는 애플리케이션 만들기

![IMG_100088CC288A-1.jpeg](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%/IMG_100088CC288A-1.jpeg)

## proposal

제목 : 종이 신문기사 이미지로부터 기사 요약 

### 1. introduction

최근 수강중인 교양수업에는 매일 종이신문을 읽고 기사를 요약해야 하는 과제가 있다. 매일 최신화가 되는 신문을 읽고 요약하는 것은 꽤 시간이 걸리는 작업으로, 머신러닝 모델을 통해 자동화될 수 있겠다는 생각이 들었다.

본 프로젝트는 종이 신문 기사 이미지로부터 디지털화된 텍스트를 생성하고, 생성된 텍스트로부터 이를 요약해주는 머신러닝 모델을 설계해보고 구현해보고자 한다. 이미지로부터 텍스트를 생성하는 컴퓨터 비전 task인 OCR(Optical Character Recoginion)과 텍스트로부터 핵심 문장을 요약해주는 자연어처리 task를 수행하는 모델을 구현 및 학습해보고, 신문이미지로부터 한 문장으로 기사를 요약해주는 end-to-end 애플리케이션을 구현하는 것이 최종 목표이다.

### 2. Problem definition & challenges

본 프로젝트의 핵심 task는 OCR모델의 구현 및 학습이다. OCR모델은 네이버 CLOVA OCR, TesseractOCR 등의 훌륭한 애플리케이션이 이미 존재하지만 단락과 열이 존재하는 신문 기사의 특성 상 완벽한 텍스트인식이 제대로 되지 않음을 확인할 수 있었다. 이를 해결하기 위해 pretrained model에  신문기사 이미지와 텍스트를 fine tuning 하는 방법과 text detection에서의 바운딩 박스의 좌표값을 이용하여 문단별로 text recognition을 하는 방법을 시도해 볼 예정이다.

예상되는 challenge는 다음과 같다. 우선 text detection / recognition / summarization 에 대한 모델들의 학습이 한정된 시간에 이루어져야 보니 학습환경과 시간에 제약을 많이 받을 것이다. 또한 사용될 머신러닝 모델들에 대한 학문적 베이스가 아직은 부족한 상태이기 때문에 이를 구현하고 활용하는데 필요한 선행 공부가 절실한 상황이다. 두 번째로는 한글 신문과 한글 텍스트, 한글 요약을 목적으로 하다보니 데이터가 상대적으로 부족하며, 데이터 전처리 과정에서 더 까다로울 것으로 생각된다. 이는 크롤링과 직접 찍은 신문 사진 등을 활용하여 해결해 볼 것 이다.

### 3. Related Works

본 프로젝트의 핵심 task인 OCR모델과 관련하여 발표된 선행연구는 다음과 같다.

- CRAFT / TPS-ResNet(Naver Clova) : 본 프로젝트의 baseline이며 공개된 pretrained-model를 fine-tuning할 것이다.
- Tessrect OCR(Google) : 오픈소스 텍스트인식 엔진
- EasyOCR : Text Detection으로 CRAFT를 사용하고, Recognition으로는 CRNN을 사용한 모델이다.

### 4. Datasets

본 프로젝트의 학습을 위해 사용할 데이터셋은 다음과 같다.

1. 한국경제신문 실물 이미지(직접 촬영), 한국경제신문 웹사이트 크롤링을 통한 텍스트 레이블
2. 한국어 텍스트 인식 학습을 위한, 한국어 인쇄체, 인쇄체 증강 데이터(from AI HUB)
3. 기사 요약을 위한 한국어 신문기사, 문서요약 데이터 (from 한국어 문서 추출요약 AI 경진대회)

오픈 소스로 공개된 데이터 외에 직접 촬영하는 데이터는 augmentation을 통해 부족한 개수를 보충한다.

[https://dacon.io/competitions/official/235671/overview/description](https://dacon.io/competitions/official/235671/overview/description/)

### 5. State-of-the-art methods and baselines

OCR에서 SOTA는 82.6\% 정확도를 기록하고 있는 MaskOCR로,  transformer 기반의 masked encoder-decoder 모델이다.  Baseline으로는 네이버에서 발표한 text detection에서는 CRAFT 모델을 이용할 예정이며, text recognition에서도 역시 네이버 Clova에서 공개한 TPS-ResNet을 이용할 예정이다.

### 6. Scedule

~ 10/16  관련 논문 공부 및 데이터 수집(상시수집)

~ 10/23  모델 시험 및 baseline 모델 학습

~ 11/20 모델 구현 및 학습

~ 11/27 애플리케이션 완성 

~ 12/5 리포트 작성 및 제출

## 데이터셋

한국어 기사 요약 : [https://dacon.io/competitions/official/235671/data](https://dacon.io/competitions/official/235671/data)

AI HUB() : [https://www.aihub.or.kr/](https://www.aihub.or.kr/)

## 개발환경

python 3.7

## pipeline

1차 목표 : 기존 모델을 활용하여 

1. input : 한글 기사 이미지 데이터셋 만들기
    1. 헤드라인 / 막 여러개로 분리 해야 하는 어려움.
2. Text Detection : 바운딩 박스 처리 → CNN을 주로 사용 (CRAFT)
    - 네이버 CRAFT
    - 관련
        
        craft 논문 : [https://arxiv.org/abs/1904.01941](https://arxiv.org/abs/1904.01941)
        
        craft 논문 리뷰 : [https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-분석-da987b32609c](https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-%EB%B6%84%EC%84%9D-da987b32609c)
        
        craft 깃헙 : [https://github.com/clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
        
    - 해야 할 일
        
        크래프트로 text detection 직접 해보기
        
3. Text Recognition :  bounding 박스가 어떤 내용인지 알아냄. → RNN/Transformer
    - 네이버 deep-text-recognition

## 참고자료

- 한국 의약품 OCR 인식 : [https://cvml.tistory.com/18](https://cvml.tistory.com/18)
- 티켓 OCR 분석 : [https://velog.io/@hyunk-go/크롤링-Tesseract-OCR-EasyOCR-OpenCV-그리고-학습](https://velog.io/@hyunk-go/%ED%81%AC%EB%A1%A4%EB%A7%81-Tesseract-OCR-EasyOCR-OpenCV-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%ED%95%99%EC%8A%B5)
- OCR 최신 동향 : [https://yongwookha.github.io/MachineLearning/2022-02-08-current-ocrs](https://yongwookha.github.io/MachineLearning/2022-02-08-current-ocrs)
- bertsum 한국어 :
    
    [https://velog.io/@raqoon886/KorBertSum-SummaryBot](https://velog.io/@raqoon886/KorBertSum-SummaryBot)
    

## 챌린지

1. CRAFT, deep-text-recognition이 얼마나 잘 되는지를 확인해야함.
2. TessractOCR
3. EasyOCR

## 깃, 주피터 명령어

- conda env list : 가상환경
- 맥os python3.6 설치 :
    
    [https://stackoverflow.com/questions/70205633/cannot-install-python-3-7-on-osx-arm64](https://stackoverflow.com/questions/70205633/cannot-install-python-3-7-on-osx-arm64)
    
- jupyter notebook kernel 추가
    
    pip install ipykernel 
    
    python -m ipykernel install --user --name 가상환경 이름 --display-name 커널 이름
    

## 개발일지

- 10/11 (화)
    1. naver Craft로 바운딩 박스 띄워보기 → 성공
        - 거의 건들지 않아도 될 수준으로 학습이 잘 되어 있음, 학습을 내가 추가적으로 시키지도 못함.
        - training code가 공개되지 않아서, fine tuning을 할 수도 없음 → 프로젝트에 딥러닝 포인트가 없을듯.
        - 만약 이걸로 하면, 바운딩 박스를 클러스터링해서 image recognition 을 하면 문제는 txt를 잘 뽑아 낼 수는 있을 듯.
        - [Error] OSError: image file is truncated : 사이즈 조절 필요함.
            
            [Craft 결과](https://www.notion.so/Craft-ef55f6352f4b48cbb64efe1e73a84411)
            
    2. tessact OCR 
        
        [EasyOCR 사용자 모델 학습하기 (1) - 시작하기 전에](https://davelogs.tistory.com/76)
        
        이 사람 블로그 많이 참고함.
        
        - 결과 : craft 보다는 상당히 구리지만, 그만큼 학습의 여지가 많이 남아있음. 학습과정도 뒤에 설명되있어서 이걸로 하면 좋을듯
        
        ![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20167b04eba1464a9da95f0f099b6bf9e0/Untitled.png)
        
        - tesseract에서 앞으로 해야 할 일
            1. 파이썬 코드 없이 C/C++ base인게 문제
            2. 코드 쪼개서 이미지에 바운딩 박스 만들어보기
            3. 모델 학습하기
            4. evaluation 생각하기
            5. **바운딩 박스 제대로 해서 단락화시켜서 텍스트화하기**
            
- 10/14 (금)
    1. Easy_Ocr test
        - 테스트 결과
            
            [easy ocr 결과](https://www.notion.so/easy-ocr-22c532b5267543a9a756cddcb9072e0f)
            
        - 꽤 정확한 수치이지만 평범한 글자임에도 아직 완벽하게 읽지는 못하고 있음.
        - 바운딩 박스 정확
    
    1. baseline과 방향 관련
        
        baseline : finetuning 하지 않은 easyocr
        
        → finetune한 이후 text recognition에서 유의미한 차이 보이기
        
        → 바운딩 박스 / 볼드체 등으로 분류해서 단락별로 텍스트 처리가 되어 확실한 차이 불러오기
        
        → 위의 방법 말고, 단락단위 학습을 시켜서 제대로 나오는지도 확인해보고 싶음.
        
    2. 데이터셋은 어떻게 할 것인가.
        - 크롤링을 해야 하나? → 단순히 한글화만 시키면 되서  딱히 필요없긴함. 아님!!
        - AI hub에서 구하기
        1. textRecognitionDataGenerator 로 만든 한국어 데이터셋
        2. 캡쳐한 이미지, 크롤링한 텍스트
            - 이미지가 크고, 텍스트도 길것으로 예상되서 학습이 잘 될지 모르겠음.
            - 2022.09.08(목) ~ 한국경제신문
                - 문장단위로 데이터셋 만들기
            - 삽입된 광고, 이미지 때문에 데이터 전처리과정에서 힘들 수 있음.
            - bounding 박스를 통해, 문단별로 할 수는 있을 듯?
                - 네이버 CRAFT / EasyOCR로
                
                ![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20167b04eba1464a9da95f0f099b6bf9e0/Untitled.jpeg)
                
                이 바운딩 박스를 어떻게 조정하.. 참..
                
                해상도 좋게 저장하는 툴 이용 plt 너무 그지같음.
                
                1. 일단 한 줄 단위로 끊기, 옆이랑 차이가 많이 나는 순간 
                2. 헤드라인 /소제목 / 본문 분류 → 세로 길이로 얘는 마지막에 하는게 좋을듯
                3. 본문 column 분류
                
                 
                
                1. 이미지나 기사가 아닌 것은 그 밑에 딸린 주석까지 전부 blur 처리해야할 듯
                2. 
                
                headline : 
                
- 10/27 (목) : bounding box 코드정리
- 11/14 (월)
    - 남은 agenda 정리
        1. bounding box 코드 모듈화 / 오류 잡기 (모든 사진에 대해서)
            
            ```sql
            jupyter notebook 키고, google.colab 설치하면 자꾸 꺼짐....-> 왜?????????
            
            ```
            
        2. 실제 한경 크롤링해서 캡쳐해서 데이터화 하기
            - 규칙 / 알고리즘 화
        3. 학습 모델 정립, 실제 학습하기 , 학습 환경 정하기
        4. 자연어처리 모델 학습하기 (80프로 와
            
            [https://younghwani.github.io/posts/kobart-summary-3/](https://younghwani.github.io/posts/kobart-summary-3/)
            
        5. 
        
- 11/27 (일)
    - Kobart를 이용한 자연어처리 모델 학습
    
    [[NLP] KoBART 요약 실행해보기(3/3)](https://younghwani.github.io/posts/kobart-summary-3/)
    
    - easyocr 해보기
- 11/28(월)
    
    일단 easyOCR 한사이클 완료
    
    easyocr보다는, deep-text-recognition-benchmark을 사용하는 게 더 나을듯.
    
    DTRB를 한글로 학습시키는게 이제 목표!
    
    DTRB를 이제 한글데이터로 학습을 시킬건데
    
    사용할 데이터
    
    1) 기존 한글자 한글자에 대한 데이터
    
    2) 바운딩박스를 유지한채 한문장~한문단이 있는 데이터 → 안귀찮으면 할 수 있을듯.
    
    원래 DTRB
    
- 11/29(화)
    
    하시발 자꾸 한글 데이터가 학습이 안됨.
    
    → 한글자 단위말고,  AI hub데이터로 여러글자 단위로 한번 해보기. 오늘 이거 마무리하고 DTRB 학습돌려놓고 자야함.
    
    ```
    python3 create_lmdb_dataset.py \
            --inputPath ../data/raw/train/ \
            --gtFile ../data/raw/train/gt.txt \
            --outputPath ../data/lmdb/train/
    ```
    
- 11/30 (수)
    1. 첫 번째 학습 완성, 결과는 개망.
        
        내일 TPS로 다시 한번 해봐야할듯.
        
    2. 데이터셋이 완성
    3. CRAFT 결과 자동화도 완성함.
    
    - [ ]  시연할거 10개정도 직접 캡쳐해서 알고리즘 확인
    - [ ]  자동화 해야함.
        1. CRAFT로 이미지 바운딩 박스 txt 따기 →
        2. 바운딩 box 문장 화 → bounding box 모듈화하기 해야돼.. 시발 귀찮아..
        3. 문장화 된 사진, 학습된 AI 모델로 넣고 결과 받아오기
    

## 학습 0 : pretrained-g2 /

- 학습 결과
    
    ```go
    --------------------------------------------------------------------------------
    dataset_root: ../lmdb/train
    opt.select_data: ['/']
    opt.batch_ratio: ['1']
    --------------------------------------------------------------------------------
    dataset_root:    ../lmdb/train	 dataset: /
    sub-directory:	/.	 num samples: 100
    num total samples of /: 100 x 1.0 (total_data_usage_ratio) = 100
    num samples of / per batch: 192 x 1.0 (batch_ratio) = 192
    --------------------------------------------------------------------------------
    Total_batch_size: 192 = 192
    --------------------------------------------------------------------------------
    dataset_root:    ../lmdb/valid	 dataset: /
    sub-directory:	/.	 num samples: 10
    --------------------------------------------------------------------------------
    No Transformation module specified
    model input parameters 32 100 20 1 256 256 1009 25 None VGG BiLSTM CTC
    loading pretrained model from ./models/korean_g2.pth
    Model:
    DataParallel(
      (module): Model(
        (FeatureExtraction): VGG_FeatureExtractor(
          (ConvNet): Sequential(
            (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(inplace=True)
            (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (7): ReLU(inplace=True)
            (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (9): ReLU(inplace=True)
            (10): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
            (11): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (13): ReLU(inplace=True)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (16): ReLU(inplace=True)
            (17): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
            (18): Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1))
            (19): ReLU(inplace=True)
          )
        )
        (AdaptiveAvgPool): AdaptiveAvgPool2d(output_size=(None, 1))
        (SequenceModeling): Sequential(
          (0): BidirectionalLSTM(
            (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
          (1): BidirectionalLSTM(
            (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
        )
        (Prediction): Linear(in_features=256, out_features=1009, bias=True)
      )
    )
    Trainable params num :  4015729
    Optimizer:
    Adadelta (
    Parameter Group 0
        eps: 1e-08
        foreach: None
        lr: 1
        maximize: False
        rho: 0.95
        weight_decay: 0
    )
    ------------ Options -------------
    exp_name: None-VGG-BiLSTM-CTC-Seed1111
    train_data: ../lmdb/train
    valid_data: ../lmdb/valid
    manualSeed: 1111
    workers: 4
    batch_size: 192
    num_iter: 300000
    valInterval: 2000
    saved_model: ./models/korean_g2.pth
    FT: True
    adam: False
    lr: 1
    beta1: 0.9
    rho: 0.95
    eps: 1e-08
    grad_clip: 5
    baiduCTC: False
    select_data: ['/']
    batch_ratio: ['1']
    total_data_usage_ratio: 1.0
    batch_max_length: 25
    imgH: 32
    imgW: 100
    rgb: False
    character:  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘
    sensitive: False
    PAD: False
    data_filtering_off: True
    Transformation: None
    FeatureExtraction: VGG
    SequenceModeling: BiLSTM
    Prediction: CTC
    num_fiducial: 20
    input_channel: 1
    output_channel: 256
    hidden_size: 256
    num_gpu: 1
    num_class: 1009
    ---------------------------------------
    
    [1/300000] Train loss: 11.57690, Valid loss: 6.35848, Elapsed_time: 1.97204
    Current_accuracy : 10.000, Current_norm_ED  : 0.00
    Best_accuracy    : 10.000, Best_norm_ED     : 0.00
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
                              | 뭣                         | 0.2273	False
                              | 적                         | 0.5558	False
                              | 별                         | 0.5035	False
                              | 월                         | 0.6674	False
                              |                           | 0.5773	True
    --------------------------------------------------------------------------------
    Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/parallel/data_parallel.py", line 166, in forward
        return self.module(*inputs[0], **kwargs[0])
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/content/drive/MyDrive/ColabNotebooks/DL_finalproject/EasyOcr/tmp/deep-text-recognition-benchmark/model.py", line 76, in forward
        visual_feature = self.FeatureExtraction(input)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/content/drive/MyDrive/ColabNotebooks/DL_finalproject/EasyOcr/tmp/deep-text-recognition-benchmark/modules/feature_extraction.py", line 28, in forward
        return self.ConvNet(input)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/container.py", line 139, in forward
        input = module(input)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/activation.py", line 98, in forward
        return F.relu(input, inplace=self.inplace)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py", line 1455, in relu
        result = torch.relu_(input)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "train.py", line 317, in <module>
        train(opt)
      File "train.py", line 153, in train
        preds = model(image, text)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/usr/local/lib/python3.7/dist-packages/torch/nn/parallel/data_parallel.py", line 169, in forward
        return self.gather(outputs, self.output_device)
      File "/usr/local/lib/python3.7/dist-packages/torch/autograd/profiler.py", line 449, in __exit__
        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
    KeyboardInterrupt
    ```
    
- 한 글자 단위 학습
- 랜덤 생성된 한글자가 class로 분류하기에는 적절하지 않음
    
    → 문장 단위 학습이 선행되야함./
    

## 학습 1 : None-VGG-BiLSTM-CTC / pretrained model  korean-g2

- CNN기반의 pretrained model korean-g2
- 3시간 3분 동안 학습
- hyperparameter
    - trainng set 10000
    - validation set 1000
    - test set 1000
    - batch size 64
    - VGG - BiLSTM - Linear
    - optimizer : Adadelta
    - pretrained model : korean-g2 (easy ocr에서 제공)
    - 1009개의 한국어 output classes
- 학습 결과 분석
    - 3시간 가량 학습을 했지만, 정확도는 3% 남짓, 문장이기 때문에 정확도를 따지는 것이 정확한 evaluation은 아니지만 그럼에도 너무나도 낮은 수치를 기록.
    - training loss가 계속해서 줄었지만, validation loss는 어느수간 0.015정도로 수렴
        - → overfitting
    - korean-g2 모델이 아마, 문장단위의 학습을 하는 모델은 아니기 때문에, 학습의 방법이 잘못되었다고 생각
    
    ![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20167b04eba1464a9da95f0f099b6bf9e0/Untitled%201.png)
    
- 학습 결과
    
    ```json
    --------------------------------------------------------------------------------
    dataset_root: ../data_aihub/lmdb/train
    opt.select_data: ['/']
    opt.batch_ratio: ['1']
    --------------------------------------------------------------------------------
    dataset_root:    ../data_aihub/lmdb/train	 dataset: /
    sub-directory:	/.	 num samples: 10000
    num total samples of /: 10000 x 1.0 (total_data_usage_ratio) = 10000
    num samples of / per batch: 64 x 1.0 (batch_ratio) = 64
    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:566: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))
    --------------------------------------------------------------------------------
    Total_batch_size: 64 = 64
    --------------------------------------------------------------------------------
    dataset_root:    ../data_aihub/lmdb/valid	 dataset: /
    sub-directory:	/.	 num samples: 1000
    --------------------------------------------------------------------------------
    No Transformation module specified
    model input parameters 32 100 20 1 256 256 1009 45 None VGG BiLSTM CTC
    loading pretrained model from ./models/korean_g2.pth
    Model:
    DataParallel(
      (module): Model(
        (FeatureExtraction): VGG_FeatureExtractor(
          (ConvNet): Sequential(
            (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(inplace=True)
            (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (7): ReLU(inplace=True)
            (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (9): ReLU(inplace=True)
            (10): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
            (11): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (13): ReLU(inplace=True)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (16): ReLU(inplace=True)
            (17): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
            (18): Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1))
            (19): ReLU(inplace=True)
          )
        )
        (AdaptiveAvgPool): AdaptiveAvgPool2d(output_size=(None, 1))
        (SequenceModeling): Sequential(
          (0): BidirectionalLSTM(
            (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
          (1): BidirectionalLSTM(
            (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
        )
        (Prediction): Linear(in_features=256, out_features=1009, bias=True)
      )
    )
    Trainable params num :  4015729
    Optimizer:
    Adadelta (
    Parameter Group 0
        eps: 1e-08
        foreach: None
        lr: 1
        maximize: False
        rho: 0.95
        weight_decay: 0
    )
    ------------ Options -------------
    exp_name: None-VGG-BiLSTM-CTC-Seed1111
    train_data: ../data_aihub/lmdb/train
    valid_data: ../data_aihub/lmdb/valid
    manualSeed: 1111
    workers: 4
    batch_size: 64
    num_iter: 300000
    valInterval: 2000
    saved_model: ./models/korean_g2.pth
    FT: True
    adam: False
    lr: 1
    beta1: 0.9
    rho: 0.95
    eps: 1e-08
    grad_clip: 5
    baiduCTC: False
    select_data: ['/']
    batch_ratio: ['1']
    total_data_usage_ratio: 1.0
    batch_max_length: 45
    imgH: 32
    imgW: 100
    rgb: False
    character:  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률르른를름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없었엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄했행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘
    sensitive: False
    PAD: False
    data_filtering_off: True
    Transformation: None
    FeatureExtraction: VGG
    SequenceModeling: BiLSTM
    Prediction: CTC
    num_fiducial: 20
    input_channel: 1
    output_channel: 256
    hidden_size: 256
    num_gpu: 1
    num_class: 1009
    ---------------------------------------
    
    [1/300000] Train loss: 0.58678, Valid loss: 0.25997, Elapsed_time: 26.32192
    Current_accuracy : 0.500, Current_norm_ED  : 0.05
    Best_accuracy    : 0.500, Best_norm_ED     : 0.05
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    한나라당의 '역공'으로 풀이된다. 김 지사측은 한나라당 차원의 의도 | '"":                      | 0.0007	False
    부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | #꾸]'g라:뭐                  | 0.0000	False
    개국 8개 법인에 파견하기로 했다. 이외에도 대한통운은 다양한 사내 | 라{B @:                    | 0.0000	False
    수의사로서 젖소의 스트레스와 건강을 관리하고 있다. 한편, 국내 우 | 7보B염 ;                    | 0.0000	False
    받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 예#:-붙필w                   | 0.0000	False
    --------------------------------------------------------------------------------
    [2000/300000] Train loss: 0.01655, Valid loss: 0.01568, Elapsed_time: 672.76087
    Current_accuracy : 2.800, Current_norm_ED  : 0.17
    Best_accuracy    : 2.800, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    무를 넘어서는 것임을 우회적으로 비판하며, 압박한 것이다. 최 장관 | 남입게 을 곤다 기 당이산 다.         | 0.0000	False
    부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 남을 일보 씨정산 확 장저산 있다.       | 0.0000	False
    의 주변 인사들의 동선을 추적하는 것으로 전해다. 검찰은 현 의원 | 소문조 바 등을 확보니 짐 마 계원다.     | 0.0000	False
    다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 딱 조면여산 확 지조 조된면 마이다.      | 0.0000	False
    영문 제목)'는 한국에서 1천100만명을 모은 '브라더후드('태극기 휘 | 입산 랜으 사망랜 보망이다.           | 0.0000	False
    --------------------------------------------------------------------------------
    [4000/300000] Train loss: 0.00014, Valid loss: 0.01557, Elapsed_time: 1306.43260
    Current_accuracy : 3.000, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 을 다조 박동발일산 을 보인보. 다.      | 0.0000	False
    있었을 것이라면서 김 전 회장의 입국설에 회의적인반응을 보다. | '을이상.법원세법면다. 등을다.         | 0.0000	False
    의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 10조 조을 지 했다.              | 0.0000	False
    비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 을 까게방이 을 확 확 형원니이다        | 0.0000	False
    학과정을 운영하고 있으며, 지난해 약 2000여 명이 신청했을 정도로 |  빛높빛마저 확 영광 이씨언이다.        | 0.0000	False
    --------------------------------------------------------------------------------
    [6000/300000] Train loss: 0.00006, Valid loss: 0.01555, Elapsed_time: 1941.30992
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 입랜 산 조 . 했.               | 0.0001	False
    김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 바원다. 면산 를을 보 있했다.         | 0.0000	False
    편적 복지를 할 수 있는데 한 의 증세 없이도 마련할 수 있다"며 "부 |  면 게 조바이 습니다.             | 0.0000	False
    자심문(영장실질심사)은 29일 오전 서울중앙지법에서 열린다. 앞서 | 입 지입 확바 확 향이다.            | 0.0000	False
    제 전반에 관한 의견을 나눈 뒤 이 같은 방안에 합의했다. 박 장관은 |  확지갈며지촉을 두지며 산 바산 있다.     | 0.0000	False
    --------------------------------------------------------------------------------
    [8000/300000] Train loss: 0.00004, Valid loss: 0.01535, Elapsed_time: 2580.99585
    Current_accuracy : 3.000, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    안중근의사모회(이사장 안응모)가 주최하고 롯데백화점이 후원하 | 빛다니망면.이각을으 바산 영사 .        | 0.0000	False
    이다.                       | 이다.                       | 0.7740	True
    있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 한총0조을 있랜조 로 있 다 .         | 0.0000	False
    기술과 장비 개발에도 크게 도움이  것으로 기대된다. 두산중공업 | 남과갈 면 지확 지다 예방이곤세다.       | 0.0000	False
     정도다. 언제나 같이 있으면 재미있고 유쾌한 사람"이라고 말했다. | 한북조열조원이 이 합조바이 북있다.       | 0.0000	False
    --------------------------------------------------------------------------------
    [10000/300000] Train loss: 0.00003, Valid loss: 0.01529, Elapsed_time: 3217.12299
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    대 의예과 1516학번 남학생 11명은 지난해 35월 학교 인근 고집 | 심다" 확 확산 확인다.             | 0.0000	False
    오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 단 박취이세 이 언당다 이된이.사다.      | 0.0000	False
    영국 폭동 와중에 페이스북에 선동 글을 올린 젊은이들에게 중형이 | 적갈을 지 바확산 확습 니다           | 0.0000	False
    흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 면 면 갈조을 보원이 다.            | 0.0000	False
    조금 오는 곳이 있습니다. 한낮 기온은 서울 4도, 대구는 8도 예상 | 남 조 다."                   | 0.0001	False
    --------------------------------------------------------------------------------
    [12000/300000] Train loss: 0.00002, Valid loss: 0.01486, Elapsed_time: 3862.04308
    Current_accuracy : 3.000, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    식이 열리는 저녁에는 체감온도가 영하 10도 안팎으로 떨어질 것으 | 갈 보원 지 로까 기 길원습다"         | 0.0000	False
    을 감안해 집행유예를 선고한다"고 밝다. 노씨는 2003년 10월께 | 을 처며안 며안으 까y 조을 바있다.      | 0.0000	False
    주민의 인권을 실질적으로 증진하고 국제적 기준에 따라 인도적 지원 | 1씨지조 산으 으 확씨당를 했 다.       | 0.0000	False
    건강하고 품위있는 노년을 위한 정책을 만들다고 말했습니다. 어르 | 조북조원 을 하 을 보 원니다.         | 0.0002	False
    하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 한과 있 면저곤까이바 다"            | 0.0000	False
    --------------------------------------------------------------------------------
    [14000/300000] Train loss: 0.00002, Valid loss: 0.01487, Elapsed_time: 4506.67012
    Current_accuracy : 3.000, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 입씨입을 확면산으보 있다.            | 0.0002	False
    의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 |  원 을 로보원니다."              | 0.0006	False
    고 있습니다. 화재 예방에 유의하야습니다. 내일은 전국에 구름 | 1로면 정이합이 을이 일이다.          | 0.0000	False
    로 정하고 추진했다. 이들 사업에 도는 사업당 200억 원을 한도로 20 | 입지 처 에 처 일 갈를다.           | 0.0000	False
    한 규정에 따라 조사위원회, 성등평등위원회, 의과대 학생상벌위원회 | 을조지며 보원 곤당 마촉을다.          | 0.0000	False
    --------------------------------------------------------------------------------
    [16000/300000] Train loss: 0.00002, Valid loss: 0.01437, Elapsed_time: 5148.48685
    Current_accuracy : 3.000, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 처와 며안 두바 북 두조광촉 니다.       | 0.0000	False
    속할 수 있지만 본인이 사표를  이상 소환 등에 응할 가능성이 적어 |  게합면찰면 사 이산 언 습니다.        | 0.0000	False
    0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 을 북면 하했다 확씨등이 이다.         | 0.0000	False
    은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 한남갈 확산기 사 확 영 획이보이다.      | 0.0000	False
    의를 자신이 뒤집어쓰지 않기 위해 3억원의 최종 종착지가 어지 지 | 남북조 면으 해 바 이기다.           | 0.0000	False
    --------------------------------------------------------------------------------
    [18000/300000] Train loss: 0.00001, Valid loss: 0.01543, Elapsed_time: 5781.69991
    Current_accuracy : 3.000, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    , 백두산 등 우리 민족의 발자취가 묻어있는 유적지도 방할 예정이 | 남바원을 이지 확 확다두조원다.         | 0.0000	False
    심을 갖고 후원하다"고 밝다.          | 심을 갖고 원하다고 밝다.            | 0.0374	False
    만 아니라 전후좌우 방향으로 미끄러지는 것을 방지하는 효과가 있다 | 바갈다 보 다두 보된이다.            | 0.0002	False
    다.                        | 다.                        | 0.5168	True
    따라 비정규직법안 강행 처리시 24일 오전 8시부터 총파업을 벌이기 | 기조면 확 짐이 격니다"             | 0.0000	False
    --------------------------------------------------------------------------------
    [20000/300000] Train loss: 0.00001, Valid loss: 0.01466, Elapsed_time: 6418.99254
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    대한 재검토와 향후대책을 논의하는 등 비상 대기상태에 들어다. | 을지며법안법안처짐n바방니있다.          | 0.0000	False
    영장을 청구했으며 긴급체포돼 수감 중인 은 전 위원을 이날 다시 불 | 입을갈지 수으 며면 조원니.           | 0.0000	False
    시 김종창 금감원장에게 청탁한 정황을 포착한 것으로 30일 확인 | 을다을 처면보 원 지사당 하기지보 다.     | 0.0000	False
    다. 감찰부는 c검사에 대해 진상규명을 위해 대검에 출석할 것을 통 | 1 망합산 을 당바이형원니.           | 0.0000	False
    업부서로 배치해 검사기능과 소비자보호 업무를 대폭 강화한다는 방 | 남 지조 지 보발일 을 " 있다.        | 0.0000	False
    --------------------------------------------------------------------------------
    [22000/300000] Train loss: 0.00001, Valid loss: 0.01511, Elapsed_time: 7054.54070
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 한과바 등 있 면저곤까이바 다"         | 0.0000	False
    수 있도록 도와달라"며 수차례 당부했으나 민노당 의원들이 거부하자 | 기조30조지 바이 로 원을다.          | 0.0000	False
    계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 빛바랜 을 지 된 발다 했다.          | 0.0011	False
    한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 을의 면 보며으 안처보로며보법면다.       | 0.0000	False
    과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 남갈을다. 습원다.                | 0.0000	False
    --------------------------------------------------------------------------------
    [24000/300000] Train loss: 0.00001, Valid loss: 0.01510, Elapsed_time: 7698.84520
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 빛을 방이면기 정 바하다고 습다.        | 0.0000	False
    올린 것에 대해서는 지나친 중형"이라고 주장했다. 반면 공공의 안녕 | 을 켜며 며처 을세다 일 보고도.        | 0.0000	False
    조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 1상조랜조 으할 을 벌이 촉다.         | 0.0000	False
    정권자는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김 | 다로 사된이 계이조 이된 있다."        | 0.0000	False
    다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 1 조원랜이 인 으로조면 있다.         | 0.0000	False
    --------------------------------------------------------------------------------
    [26000/300000] Train loss: 0.00001, Valid loss: 0.01480, Elapsed_time: 8331.05842
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    은 이날 요코하마의 파시피코홀에서 자선행사 '28day'n year | 랜갈 으당 망이 기망이 .하다.         | 0.0000	False
    정이다. 은행권역에서는 은행감독국이 신설되고 외환업무실이 외환 | 기 보방에 두곤다. 다호 형향다.        | 0.0000	False
    문제 해결과 한반도 평화를 일구도록 노력하다고 약속했습니다. 지 | 입 확면지법 방법 면 며치안 보며.       | 0.0000	False
    의 대표주자로 국내에서도 인기를 누리고 있으며, 각종 와인 평가에 | 지 면조원할 지보이 등원니.           | 0.0000	False
    의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 10조 조을 1조원 했다.            | 0.0000	False
    --------------------------------------------------------------------------------
    [28000/300000] Train loss: 0.00001, Valid loss: 0.01530, Elapsed_time: 8961.74624
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 단 면며로 여산을이다 고를을 이다.       | 0.0000	False
    사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 지난을 발산 짐마지 " 에보입이다.       | 0.0000	False
    입을 열면 소환시기를 앞당길 계획이다.     | 입을 열면 소환시기를 앞당길 획이다.      | 0.4602	False
    니다. 현재 중부 지방과 영남을 중심으로 건조 특보가 확대되고 있습 | 한북지조 며 을확 3t고 획이.다.       | 0.0000	False
    j그룹 회장에게 그 뜻을 전달했다고 진술했습니다. kbs 뉴스 이현기 | 로특합이지 등두 두다 앞 조짐마보니다.     | 0.0000	False
    --------------------------------------------------------------------------------
    [30000/300000] Train loss: 0.00001, Valid loss: 0.01511, Elapsed_time: 9592.40041
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    동강 단관로 사고시설 파손우려가 있다고 통보했고, 수회에 걸쳐 | 빛씨정일 보사망하다 기 저 바이다.       | 0.0000	False
    둥 번개가 치습니다. 비는 밤에 서쪽 지역부터 차차 그치습니다. | 한원향방상을확에적보이에 바 있다"        | 0.0000	False
    빛바랜 영광입니다.ytn 이형원입니다.     | 빛바랜 영광입니다ytn 형원입니다.       | 0.0613	False
    와의 동일여부 등을 확인할 예정이다.      | 와의 동일여부 등을 확인할 예정이다.      | 0.2717	True
    파도가 높고 날이 어두워지자 이날 구조작업은 철수하고 17일 오전 | 0지게법처 곤두기을안 하을 "있다.       | 0.0000	False
    --------------------------------------------------------------------------------
    [32000/300000] Train loss: 0.00001, Valid loss: 0.01457, Elapsed_time: 10222.17637
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    동 자택과 1억 원짜리 수표 30장 등 모두 58억 원 규몹니다. cj그룹 |  며바며 으 두 곤세다.             | 0.0000	False
    유동업조합으로 시작한 서울우유는 이후 국내 낙농산업의 신을 선 | 지정0으 를 부. 취지 갈조있다.        | 0.0000	False
    3건 25억7000여만원을 삭감했다. 여기에는 김 지사의 대표적인 공 | 기다북조격을 확찰이 정원이 이도다.       | 0.0000	False
    후배가 되는 것을 싫어하는 것을 보고 아내의 지인이 사는 강동구로 | 적 구 를 확 세할 조. 다.          | 0.0002	False
    자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 적 랜 확다 촉조두이 등으보 습니다.      | 0.0000	False
    --------------------------------------------------------------------------------
    [34000/300000] Train loss: 0.00001, Valid loss: 0.01476, Elapsed_time: 10851.70333
    Current_accuracy : 2.900, Current_norm_ED  : 0.17
    Best_accuracy    : 3.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    개그맨 김형곤이 다이어트 테마 섬 개발을 위해무인도를 구입했다는 | 빛영정며 지 보n 습 보바니다.         | 0.0000	False
    비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 을로 산 바 니 다.               | 0.0003	False
    러나면서 현 전 의원이 종착지인지 여부를 밝히는 데 조씨의 진술이 | 안 며산 두. 를일까하 지 .          | 0.0000	False
    다. 이 관계자는 이어 "전입학 당시 오 교사가 전입학서류를 담당 교 | 적이0다이기정보하산 면 저 촉격다"       | 0.0000	False
    표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 한 향방 을 보산 두산 확면를을.        | 0.0000	False
    --------------------------------------------------------------------------------
    ```
    

## 학습 2: TPS-ResNet-BiLSTM-CTC

- 학습 결과 분석
    - 2시간 가량의 학습, 정확도는 3% 남짓
    - 단어 길이의 짧은 문장은 맞추는 것으로 보아 학습 상태가 엉망은 아님.
    - 가면 갈 수록 ‘기자’ 로 끝나는 prediction이 많아 지는 것으로 보아 overfitting
    - classfication의 CTC가 parameter 가 너무 작은 것으로 판단. attention 모델로 돌려봄
    - Trainable params num :  49107897
    
    ![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20167b04eba1464a9da95f0f099b6bf9e0/Untitled%202.png)
    
- 학습 결과

```go
--------------------------------------------------------------------------------
dataset_root: ../data_aihub/lmdb/train
opt.select_data: ['/']
opt.batch_ratio: ['1']
--------------------------------------------------------------------------------
dataset_root:    ../data_aihub/lmdb/train	 dataset: /
sub-directory:	/.	 num samples: 10000
num total samples of /: 10000 x 1.0 (total_data_usage_ratio) = 10000
num samples of / per batch: 64 x 1.0 (batch_ratio) = 64
--------------------------------------------------------------------------------
Total_batch_size: 64 = 64
--------------------------------------------------------------------------------
dataset_root:    ../data_aihub/lmdb/valid	 dataset: /
sub-directory:	/.	 num samples: 1000
--------------------------------------------------------------------------------
model input parameters 32 100 20 1 512 256 1009 45 TPS ResNet BiLSTM CTC
Skip Transformation.LocalizationNetwork.localization_fc2.weight as it is already initialized
Skip Transformation.LocalizationNetwork.localization_fc2.bias as it is already initialized
Model:
DataParallel(
  (module): Model(
    (Transformation): TPS_SpatialTransformerNetwork(
      (LocalizationNetwork): LocalizationNetwork(
        (conv): Sequential(
          (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (10): ReLU(inplace=True)
          (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (12): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (13): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (14): ReLU(inplace=True)
          (15): AdaptiveAvgPool2d(output_size=1)
        )
        (localization_fc1): Sequential(
          (0): Linear(in_features=512, out_features=256, bias=True)
          (1): ReLU(inplace=True)
        )
        (localization_fc2): Linear(in_features=256, out_features=40, bias=True)
      )
      (GridGenerator): GridGenerator()
    )
    (FeatureExtraction): ResNet_FeatureExtractor(
      (ConvNet): ResNet(
        (conv0_1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn0_1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv0_2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn0_2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (layer1): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (layer2): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (maxpool3): MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
        (layer3): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (3): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (4): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (conv3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (layer4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (conv4_1): Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), bias=False)
        (bn4_1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv4_2): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1), bias=False)
        (bn4_2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (AdaptiveAvgPool): AdaptiveAvgPool2d(output_size=(None, 1))
    (SequenceModeling): Sequential(
      (0): BidirectionalLSTM(
        (rnn): LSTM(512, 256, batch_first=True, bidirectional=True)
        (linear): Linear(in_features=512, out_features=256, bias=True)
      )
      (1): BidirectionalLSTM(
        (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
        (linear): Linear(in_features=512, out_features=256, bias=True)
      )
    )
    (Prediction): Linear(in_features=256, out_features=1009, bias=True)
  )
)
Trainable params num :  49107897
Optimizer:
Adadelta (
Parameter Group 0
    eps: 1e-08
    foreach: None
    lr: 1
    maximize: False
    rho: 0.95
    weight_decay: 0
)
------------ Options -------------
exp_name: TPS-ResNet-BiLSTM-CTC-Seed1111
train_data: ../data_aihub/lmdb/train
valid_data: ../data_aihub/lmdb/valid
manualSeed: 1111
workers: 4
batch_size: 64
num_iter: 100000
valInterval: 100
saved_model: 
FT: False
adam: False
lr: 1
beta1: 0.9
rho: 0.95
eps: 1e-08
grad_clip: 5
baiduCTC: False
select_data: ['/']
batch_ratio: ['1']
total_data_usage_ratio: 1.0
batch_max_length: 45
imgH: 32
imgW: 100
rgb: False
character:  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률르른를름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없었엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄했행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘
sensitive: False
PAD: False
data_filtering_off: True
Transformation: TPS
FeatureExtraction: ResNet
SequenceModeling: BiLSTM
Prediction: CTC
num_fiducial: 20
input_channel: 1
output_channel: 512
hidden_size: 256
num_gpu: 1
num_class: 1009
---------------------------------------

[1/100000] Train loss: 1.52985, Valid loss: 0.96108, Elapsed_time: 11.10016
Current_accuracy : 0.000, Current_norm_ED  : 0.00
Best_accuracy    : 0.000, Best_norm_ED     : 0.00
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한인권법을 놓고 하게 맞서는 동안 북한은 이같은 움직임에 대한 | 랑옛껴니R흑R니흑쭉멈쭉              | 0.0000	False
조합 조합장은 "서울우유가 75년간 업계 1위 자리를 지켜올 수 있었 | 랑방니껴니껴J껴니껴본니짚             | 0.0000	False
동 자택과 1억 원짜리 수표 30장 등 모두 58억 원 규몹니다. cj그룹 | 랑옛껴 R니껴R본멈짚               | 0.0000	False
수 있도록 도와달라"며 수차례 당부했으나 민노당 의원들이 거부하자 | 랑옛껴흡껴R특R흑R흑본컨흑멈           | 0.0000	False
님으로 와 알게된 정모씨에게 부탁해 c군의 주소를 명일동 소재 정씨 | 랑니껴니R니흑본짚멈                | 0.0000	False
--------------------------------------------------------------------------------
[100/100000] Train loss: 0.25937, Valid loss: 0.17976, Elapsed_time: 113.55976
Current_accuracy : 0.100, Current_norm_ED  : 0.06
Best_accuracy    : 0.100, Best_norm_ED     : 0.06
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
있는 반불이를 주제로 한 문화관광 우수축제인 '제15회 무주반 |   .                       | 0.0000	False
입니다.                      | 다.                        | 0.0035	False
사용할 계획이다. 이재우 사장은 "이번 협약 체결로 병마와 싸우고 |  다.                       | 0.0000	False
해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 |  다.                       | 0.0000	False
임직원들로부터 높은 인기를 얻고 있다. 영어, 중국어, 일본어 외에도 |  다.                       | 0.0000	False
--------------------------------------------------------------------------------
[200/100000] Train loss: 0.18049, Valid loss: 0.17311, Elapsed_time: 150.89041
Current_accuracy : 0.100, Current_norm_ED  : 0.06
Best_accuracy    : 0.100, Best_norm_ED     : 0.06
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
입장을 발표하고, 사회적으로 손가락질받을 일이 학교에서 발생해 안 |  다.                       | 0.0000	False
편적 복지를 할 수 있는데 한 의 증세 없이도 마련할 수 있다"며 "부 |  다.                       | 0.0000	False
끼치고 있다. 하지만 수자원공사는 공기업 경영평가에서는 작성기준 |  다.                       | 0.0000	False
지거나 이러한 수분이 제대로 배출되지 못하면 차가 미끄러지게 된다 |  다.                       | 0.0000	False
받고 있다. 권 수석은 전화 통화를 인정했고 "일언지하에 거절했다"고 |  다.                       | 0.0000	False
--------------------------------------------------------------------------------
[300/100000] Train loss: 0.16011, Valid loss: 0.13982, Elapsed_time: 186.04374
Current_accuracy : 0.100, Current_norm_ED  : 0.07
Best_accuracy    : 0.100, Best_norm_ED     : 0.07
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
편적 복지를 할 수 있는데 한 의 증세 없이도 마련할 수 있다"며 "부 |  다.                       | 0.0000	False
무를 넘어서는 것임을 우회적으로 비판하며, 압박한 것이다. 최 장관 |  다.                       | 0.0000	False
우유, 행복한 고객'을 실현하다고 선언했다. 75년을 넘어 100년으 |  다.                       | 0.0000	False
과 오씨가 서로 아는 사이라는 사실도 뒤늦게 우연히 알게다"고 덧 |  다.                       | 0.0000	False
건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 |  다.                       | 0.0000	False
--------------------------------------------------------------------------------
[400/100000] Train loss: 0.13708, Valid loss: 0.12552, Elapsed_time: 221.79814
Current_accuracy : 0.100, Current_norm_ED  : 0.07
Best_accuracy    : 0.100, Best_norm_ED     : 0.07
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
선고다. 글랜드 체스터 지방법원이 페이스북에 폭동을 선동하는 |  니다.                      | 0.0000	False
의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 |  니다.                      | 0.0000	False
발견, 경찰에 신고 했다. 이씨는 경찰에서 "집에 돌아와 보니 함께 지 |  다.                       | 0.0000	False
단과 총격을 벌이다 경찰이 사망하기도 했다.  |  니다.                      | 0.0000	False
군산시에서는 5월 4일부터 8일까지 5월의 보리밭, 추억속으로 안내 |  다.                       | 0.0000	False
--------------------------------------------------------------------------------
[500/100000] Train loss: 0.12115, Valid loss: 0.10939, Elapsed_time: 258.00869
Current_accuracy : 0.100, Current_norm_ED  : 0.08
Best_accuracy    : 0.100, Best_norm_ED     : 0.08
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 의 다.                      | 0.0000	False
다. 한국정신대문제대책협의회 안정미 팀장은 "일본군 위안부 피해자 | 의 다.                      | 0.0000	False
연두색으로 표시된 많은 지역도 강풍 주의보가 발효중인데요. 실제로 |  니다.                      | 0.0000	False
롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 의 다.                      | 0.0000	False
한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 의 니다.                     | 0.0000	False
--------------------------------------------------------------------------------
[600/100000] Train loss: 0.11464, Valid loss: 0.11075, Elapsed_time: 293.03312
Current_accuracy : 0.100, Current_norm_ED  : 0.08
Best_accuracy    : 0.100, Best_norm_ED     : 0.08
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  |  다 다.                     | 0.0000	False
시한 결과다. 경북도와 구미시는 물론 가물막이 아래 준설공사를 담 |  다.                       | 0.0000	False
다. 한국정신대문제대책협의회 안정미 팀장은 "일본군 위안부 피해자 | 을 상정보 기"                  | 0.0000	False
건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 |  다.                       | 0.0000	False
부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 |  정보니다.                    | 0.0000	False
--------------------------------------------------------------------------------
[700/100000] Train loss: 0.10793, Valid loss: 0.11201, Elapsed_time: 328.89987
Current_accuracy : 0.100, Current_norm_ED  : 0.08
Best_accuracy    : 0.100, Best_norm_ED     : 0.08
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
다. 감찰부는 c검사에 대해 진상규명을 위해 대검에 출석할 것을 통 | 기 을 이다.                   | 0.0000	False
족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 대다.                       | 0.0000	False
을 수 있도록 하는 것이다. 이 의원은 "최근 일본군 위안부 피해자들 | 을을 다.                     | 0.0000	False
크 서비스를 규제하는 방안을 검토하다고 밝다. 데일리직 길인 | 을 을 다.                    | 0.0000	False
한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 을 에 다.                    | 0.0000	False
--------------------------------------------------------------------------------
[800/100000] Train loss: 0.09158, Valid loss: 0.08750, Elapsed_time: 364.72710
Current_accuracy : 0.300, Current_norm_ED  : 0.09
Best_accuracy    : 0.300, Best_norm_ED     : 0.09
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
서의 역할이라고 밝히고 있다. 검찰은 사실관계를 확인하기 위해 박 | 을 있다.                     | 0.0000	False
화축제가 전주시 풍남동 경기전 일대에서 열린다. 임실군에선 30일 | 성 0 니다.                   | 0.0000	False
임원급 인사와 함께 발표할 예정이다. 개편안에 따르면 기존 감독서 | 다. 기.                     | 0.0000	False
이미경 부회장 퇴진을 강요한 의를 받는 조원동 전 청와대 경제수 | 단 을 다.                    | 0.0000	False
럭으로 길목을 차단해 경찰의 접근을 막으며 3대의 픽업과 1대의 | 을 다.                      | 0.0000	False
--------------------------------------------------------------------------------
[900/100000] Train loss: 0.07480, Valid loss: 0.08540, Elapsed_time: 400.57592
Current_accuracy : 0.400, Current_norm_ED  : 0.11
Best_accuracy    : 0.400, Best_norm_ED     : 0.11
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
로 고용했으며 로르 회장은 2003년 이래 김 전회장을 최소 세 번 만 | 단 0 원니다.                  | 0.0000	False
제 상황처럼 해볼 수 있게 돼정비 전문인력 양성은 물론, 새로운 정비 | 단 인다.                     | 0.0000	False
사진을 위주로, 최근 일본 개봉 스줄이 나온 원 주연의 '우리형' | 의 다자                      | 0.0000	False
대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 의 이다.                     | 0.0000	False
감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 한 을 다.                    | 0.0000	False
--------------------------------------------------------------------------------
[1000/100000] Train loss: 0.06864, Valid loss: 0.09884, Elapsed_time: 437.40946
Current_accuracy : 0.100, Current_norm_ED  : 0.10
Best_accuracy    : 0.400, Best_norm_ED     : 0.11
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
조금 오는 곳이 있습니다. 한낮 기온은 서울 4도, 대구는 8도 예상 | 의 기.자                     | 0.0000	False
의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 의 다.                      | 0.0000	False
금감독원이 금서비스개선국을 신설하고 외환업무실을 외환감독 | 의 을보 니다.                  | 0.0000	False
장경기록문화 테마파크 조성, 통영 국제음악당 건립, 김해 중소기업 | 적 을 다.                    | 0.0000	False
한국광해관리공단(이사장 권인)은 국내뿐만 아니라 세계 각 국별로 | 의 다.                      | 0.0000	False
--------------------------------------------------------------------------------
[1100/100000] Train loss: 0.05969, Valid loss: 0.06063, Elapsed_time: 472.78829
Current_accuracy : 0.200, Current_norm_ED  : 0.12
Best_accuracy    : 0.400, Best_norm_ED     : 0.12
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 한다" 으 니다."                | 0.0000	False
영문 제목)'는 한국에서 1천100만명을 모은 '브라더후드('태극기 휘 | 성의 켜며처 리을습니다.             | 0.0000	False
초 저질던 범죄보다 큰 사회적 비용이 들게 된다"고 말했다. 시민단 | 기을 원 했다.                  | 0.0000	False
성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 한문 이했다.                   | 0.0000	False
비정규직법안에 대한 국회 처리를 둘러싸고 여당과 노동계가 한 | 의 지 을 했다.                 | 0.0000	False
--------------------------------------------------------------------------------
[1200/100000] Train loss: 0.04029, Valid loss: 0.04190, Elapsed_time: 508.59997
Current_accuracy : 0.900, Current_norm_ED  : 0.13
Best_accuracy    : 0.900, Best_norm_ED     : 0.13
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
입을 비롯한 다양한 의혹이제기에 따라 이달 20일부터 c검사에 대 | 을 열 했다.                   | 0.0000	False
으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 성마 자를 위해고 기자              | 0.0000	False
한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 을 보법 기자                   | 0.0000	False
유효슈팅 없이 0대 1로 완패한 데 이어, 6차전에서 다시 만난 중국에 | 공 열린 0이직 기자               | 0.0000	False
별법 적용을 받게 니다. 조원동 전 청와대 경제수석에 대한 재판은 | 공 을 이원고이했다.               | 0.0000	False
--------------------------------------------------------------------------------
[1300/100000] Train loss: 0.03347, Valid loss: 0.03459, Elapsed_time: 545.27920
Current_accuracy : 1.200, Current_norm_ED  : 0.13
Best_accuracy    : 1.200, Best_norm_ED     : 0.13
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
고 소개했다.                   | 고소다.                      | 0.0196	False
것"이라며 "이 두 가지 임무에 가시적인 진전이 있기를 희망한다"고 | 성마 니다이 기직 했다.             | 0.0000	False
낮부터 기온이 오름세를 보이면서 추위가 금세 누그러지습니다. 그 | 을마 자 니1 0만을 니다.           | 0.0000	False
화를 추진하다고 밝 습니다. 평창 올림픽에 대해서는 88 올림픽 | 을열 열린 기자                  | 0.0000	False
명했다. 답안지 대리작성도 지난해 1학기 중간고사 이전부터 미리 공 | 성마 니 이 기.                 | 0.0000	False
--------------------------------------------------------------------------------
[1400/100000] Train loss: 0.02726, Valid loss: 0.03196, Elapsed_time: 581.39811
Current_accuracy : 1.800, Current_norm_ED  : 0.14
Best_accuracy    : 1.800, Best_norm_ED     : 0.14
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
빛바랜 영광입니다.ytn 이형원입니다.     | 빛랜 영입니다. 이원입니다.           | 0.0000	False
채용할 경우 증원인력 1인당 월 50만원(올해 60만원)씩한시적으로 | 북갈로확 보고 있다.               | 0.0000	False
불 축제'를 연다.                | 불 축제'를 했다.                | 0.0014	False
0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 을 지서다된 두세다."              | 0.0000	False
성가족위원회 소속 이자스민 의원은 '일본군 위안부 피해자들에게 법 | 공딱지데텔에 를을 획 있다.           | 0.0000	False
--------------------------------------------------------------------------------
[1500/100000] Train loss: 0.01662, Valid loss: 0.02036, Elapsed_time: 619.16713
Current_accuracy : 2.500, Current_norm_ED  : 0.14
Best_accuracy    : 2.500, Best_norm_ED     : 0.14
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
연합뉴스와의 전화 통화를 통해 "최근 변호사인 친동생(김형진 씨)이 | 심로 갖로 로했다.                | 0.0000	False
많습니다. 남해안은 새벽부터 아침 사이에 눈 또는 비가 내리습 | 성빛랜 광 1 만 다.              | 0.0000	False
부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 단과 총격해 0 했다.              | 0.0000	False
수 있게 하고, 고객들은 위생처리를 거친 우유를 안심하고 마실 수 있 | 단 을 속했다.                  | 0.0000	False
성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 공 켜지안 에 을 다.              | 0.0000	False
--------------------------------------------------------------------------------
[1600/100000] Train loss: 0.01507, Valid loss: 0.01811, Elapsed_time: 656.49557
Current_accuracy : 2.800, Current_norm_ED  : 0.15
Best_accuracy    : 2.800, Best_norm_ED     : 0.15
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
전북도를 찾는 관광객에게 오랫동안 기억에 남을 추억거리를 선물할 | 남등을다 촉 기자                 | 0.0000	False
로 가는 중심가치를 '행복'으로 설정한 것이다. 송용헌 서울우유협동 | 날해보 기자                    | 0.0000	False
선고다. 글랜드 체스터 지방법원이 페이스북에 폭동을 선동하는 | 와마 격위 린 씨각 기자             | 0.0000	False
영문 제목)'는 한국에서 1천100만명을 모은 '브라더후드('태극기 휘 | 공 켜며호에 리에 다자              | 0.0000	False
이 과제가 국가 추진과제로 선정다. 이에 따라 오는 2014년을 목 | 와랜 영광 원 다.                | 0.0000	False
--------------------------------------------------------------------------------
[1700/100000] Train loss: 0.00678, Valid loss: 0.01088, Elapsed_time: 692.63278
Current_accuracy : 3.100, Current_norm_ED  : 0.16
Best_accuracy    : 3.100, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 의문 확 조원된 기자               | 0.0000	False
몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 을 지켜며법 처에 촉을 있다.          | 0.0000	False
성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 성 자일광 니 원니했다.             | 0.0000	False
롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 남북갈등보 지 산조을 고보 인다.        | 0.0000	False
발견, 경찰에 신고 했다. 이씨는 경찰에서 "집에 돌아와 보니 함께 지 | 기 갈입 조열이 사원 기.            | 0.0000	False
--------------------------------------------------------------------------------
[1800/100000] Train loss: 0.00706, Valid loss: 0.01448, Elapsed_time: 729.43955
Current_accuracy : 3.200, Current_norm_ED  : 0.15
Best_accuracy    : 3.200, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 공 켜호법에 에 바 했다.            | 0.0000	False
건강하고 품위있는 노년을 위한 정책을 만들다고 말했습니다. 어르 | 와 동일위 1 다."               | 0.0000	False
시키기 위해 위장전입한 것은 사실이나 오씨의 도움을 받아 b고교에 | 성 켜보며법에 린. 보 니다."         | 0.0000	False
님으로 와 알게된 정모씨에게 부탁해 c군의 주소를 명일동 소재 정씨 | 공 지 린 으보 ."               | 0.0000	False
장을 향해 또다시 목소리를 높다. 최 장관은 30일 기자간담회에서 | 적마비 을 계이 이다.              | 0.0000	False
--------------------------------------------------------------------------------
[1900/100000] Train loss: 0.00547, Valid loss: 0.00894, Elapsed_time: 765.73184
Current_accuracy : 3.400, Current_norm_ED  : 0.16
Best_accuracy    : 3.400, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
의 주변 인사들의 동선을 추적하는 것으로 전해다. 검찰은 현 의원 | 와 동여부 을 한 기자              | 0.0000	False
이들 중 15학번 남학생 9명은 주점에 후배 남학생들을 불러 동료 여 | 을 호서 열린 리직 기자             | 0.0000	False
사 등 다른 사안과 연계 가능성까지 언급해 민주당의 반발을 고, 북 | 입을 지 조짐 리 있다.             | 0.0000	False
내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 공 켜데법에 1리 각을 두다.          | 0.0000	False
이유라도 합리화수 없는 반인적 행위"라며 "그러나 극도의 경제 | 와 동격등 저 기자                | 0.0000	False
--------------------------------------------------------------------------------
[2000/100000] Train loss: 0.00259, Valid loss: 0.00738, Elapsed_time: 802.02175
Current_accuracy : 3.400, Current_norm_ED  : 0.16
Best_accuracy    : 3.400, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
입니다.                      | 니다.                       | 0.4862	False
입니다.                      | 니다.                       | 0.6402	False
속할 수 있지만 본인이 사표를  이상 소환 등에 응할 가능성이 적어 | 공 롯호에 에 촉각 두다.            | 0.0000	False
9월 중국전 한 골 차 진땀승을 시작으로, 조 최약체로 꼽히던 시리아 | 의문 지난시 확산된 것장보습니다.        | 0.0000	False
상태로 이뤄지는 콜드체인시스템을 도입해 2000여 곳의 전용목장에 | 성마 총격을해 기자                | 0.0000	False
--------------------------------------------------------------------------------
[2100/100000] Train loss: 0.00213, Valid loss: 0.00583, Elapsed_time: 838.88650
Current_accuracy : 3.600, Current_norm_ED  : 0.16
Best_accuracy    : 3.600, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 | 와 일 등을 저 기.               | 0.0000	False
과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 입을로지 산조짐 보 기자             | 0.0000	False
위반)로 김모(18)군을 불구속 입건했다. 경찰에 따르면 김군은 고교 | 성마비환 다 사정보습니다."           | 0.0000	False
니다. 현재 중부 지방과 영남을 중심으로 건조 특보가 확대되고 있습 | 와 켜보일법 다.                 | 0.0000	False
화축제가 전주시 풍남동 경기전 일대에서 열린다. 임실군에선 30일 | 성마비 자를위 0이 다.             | 0.0000	False
--------------------------------------------------------------------------------
[2200/100000] Train loss: 0.00177, Valid loss: 0.00505, Elapsed_time: 876.75768
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 단 총을 난 박 를당한 바 있다.        | 0.0000	False
다"고 벼르고 있다. 민주당은 "한나라당 법안은 보수단체 지원내용만 | 의 지광 n이 이다.               | 0.0000	False
어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 | 와 동일에 각를  기자              | 0.0000	False
다. 서울우유는 지난 11일 75주년 기념식에서 '행복한 젖소, 행복한 | 10원을 지 짐마 이 기.            | 0.0000	False
3건 25억7000여만원을 삭감했다. 여기에는 김 지사의 대표적인 공 | 성마 환 을입니다 사원형 있다.         | 0.0000	False
--------------------------------------------------------------------------------
[2300/100000] Train loss: 0.00130, Valid loss: 0.00490, Elapsed_time: 912.67146
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
께 3개 국으로 늘어나게 된다. 저축은행 검사는 1, 2국으로 나다. | 단다 총을 지최 조길 다.            | 0.0000	False
군산시에서는 5월 4일부터 8일까지 5월의 보리밭, 추억속으로 안내 | 성마 자를광위 10 0원 ."          | 0.0000	False
하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 을 켜일법처 을  있다.             | 0.0000	False
에서 개봉한다. 일본 산이스포츠는 21일 "'마이 브라더('우리형'의 | 와 총격 을 y 망이고 했다.          | 0.0000	False
파클을 함께 마시면 입안에서 청량감이 살아나 더운 날씨에 처진 | 남을로지면 30조이 이 입 니다.        | 0.0000	False
--------------------------------------------------------------------------------
[2400/100000] Train loss: 0.00124, Valid loss: 0.00455, Elapsed_time: 948.36061
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
이는 방식을 택해 공장식 축산을 할 수 없다며 한정생산을 강점으로 | 1북총격을해치 조짐 이전이다.          | 0.0000	False
부채 문제 등 취약요인에 대한 적극적인 대응이 필요한 시점이라는 | 남북등입y 조짐 이고 있다.           | 0.0000	False
금실무협의회 이후 십수년만에 부활되는 것이다. 하지만 재정부가 | 입을 열면시 앞당저 계 있다.          | 0.0000	False
연두색으로 표시된 많은 지역도 강풍 주의보가 발효중인데요. 실제로 | 소해 언를 다."                 | 0.0000	False
아들 c군은 주소지의 학교를 피하기 위해 서울 강동구로 위장전입한 | 의문 지 확세1 박장각을 인다.         | 0.0000	False
--------------------------------------------------------------------------------
[2500/100000] Train loss: 0.00101, Valid loss: 0.00420, Elapsed_time: 983.33313
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
서울우유가 치열한 유업체들간 경쟁시장에 '행복' 발을 들고 나 | 의 지2 1 박각을 다.             | 0.0000	False
오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 성 동일 을다.                  | 0.0004	False
금실무협의회 이후 십수년만에 부활되는 것이다. 하지만 재정부가 | 입 열 린 직한  기자              | 0.0000	False
생활고에 시달리다 남편과 동반자살을 하기로 한 뒤 혼자남는 딸이 | 성 영광입 린 사 직보 다."          | 0.0000	False
습니다. 오늘 경북 상주는 낮 기온 25도까지 올고 서울도 22.1도 | 성이 보 있다.                  | 0.0000	False
--------------------------------------------------------------------------------
[2600/100000] Train loss: 0.00086, Valid loss: 0.00424, Elapsed_time: 1019.06837
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
재수준의 회복 흐름을 지속하고 있으나 대외적으로 국제원자재가격 | 단과 총을 벌이다 경 사원병 했다.       | 0.0000	False
으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 을 지보 해1 0정을 기"            | 0.0000	False
역으로 지목하고 있는 '다이아드 아일랜드'에 포함돼 있어 이 같은 | 딱 해지 시발 을 바 다."           | 0.0000	False
성가족위원회 소속 이자스민 의원은 '일본군 위안부 피해자들에게 법 | 공 롯데호에 촉각을 있다.            | 0.0000	False
맞는 이 축제는 공음면 학원농장의 보리밭에서 5월 8일까지 열린다. | 와 총격등을 갖인  다.             | 0.0000	False
--------------------------------------------------------------------------------
[2700/100000] Train loss: 0.00549, Valid loss: 0.02593, Elapsed_time: 1054.69564
Current_accuracy : 2.200, Current_norm_ED  : 0.15
Best_accuracy    : 3.800, Best_norm_ED     : 0.16
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 와과 동를 확다 전 보 했다.          | 0.0000	False
를 위해 2학기부터 수업을 분리, 운영할 계획이라고 했다. 이와 함께 | 와의 일 해을습 하 기자             | 0.0000	False
원은 옛 삼성동 자택을 67억 원에 팔면서 생긴 돈으로 파악하고 있습 | 남북갈등을 경저 사것전보호 다자         | 0.0000	False
한국광해관리공단(이사장 권인)은 국내뿐만 아니라 세계 각 국별로 | 공 켜호법 에 저 전고 기.           | 0.0000	False
건축학과를 졸업하고 현재중공업분야에서 해외 수출 업무를 전담하 | 단과총격 산된 것전 다자             | 0.0000	False
--------------------------------------------------------------------------------
[2800/100000] Train loss: 0.03358, Valid loss: 0.01800, Elapsed_time: 1089.75871
Current_accuracy : 2.300, Current_norm_ED  : 0.17
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 을 광 을니다"이다.               | 0.0000	False
러나 학교측은 당시 대부분의 학급이 정원 수준인 35, 36명이었으나 | 을 동일법 에 다" 밝다.            | 0.0000	False
타깝다며 인성 교육을 더욱 강화하다고 밝다.학교 측은 이번 성 | 성마 영 을 이다 원 보했다.          | 0.0000	False
송수진입니다."                  | 송수진입니다."                  | 0.1277	True
공동으로 ks 제정 및 국제표준화(iso)를 추진한다고 9일 밝다. 광 | 의 지 1 당한 다.               | 0.0000	False
--------------------------------------------------------------------------------
[2900/100000] Train loss: 0.00771, Valid loss: 0.00822, Elapsed_time: 1126.14906
Current_accuracy : 3.600, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
비정규직법안에 대한 국회 처리를 둘러싸고 여당과 노동계가 한 | 딱 롯호 특을습당 기자              | 0.0000	False
국정원 뇌물 사건을 맡은 유영하 변호사가 보관하고 있습니다. 유 변 | 성마 환광 을 니다.               | 0.0000	False
는 23일 법안심사소위를 열어 정부가 제출한 비정규직법안을 심사해 | 을 롯며입 린 저이 형입니다.          | 0.0000	False
다. 이 관계자는 이어 "전입학 당시 오 교사가 전입학서류를 담당 교 | 공 켜보호텔 열다. 기직보종 했다.       | 0.0000	False
0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 소로 지 일박 드습속세다.            | 0.0000	False
--------------------------------------------------------------------------------
[3000/100000] Train loss: 0.00530, Valid loss: 0.00621, Elapsed_time: 1161.88932
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
급을 받기 때문이다. 하지만 감사원 감사결과 이는 평가기준과 다 | 입문이 열다 데리 오호 다.           | 0.0000	False
사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 공 지보지 0 0방직 입니다.          | 0.0000	False
수의사로서 젖소의 스트레스와 건강을 관리하고 있다. 한편, 국내 우 | 날씨정보습 전원저 전이다.            | 0.0000	False
국으로 확대 개편하는 등 조직을 대폭 확대한다. 특히 감독과 검사업 | 딱딱지 특성을 대 인다.             | 0.0000	False
뒤 취한 조니다. 법원에 청구한 동결 대상 재산은 28억 원에 매입한 | 기씨정보 니다.                  | 0.0000	False
--------------------------------------------------------------------------------
[3100/100000] Train loss: 0.01024, Valid loss: 0.03522, Elapsed_time: 1196.96796
Current_accuracy : 1.500, Current_norm_ED  : 0.14
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
니다. 앞으로 대기는 점점 더 건조해 질 것으로 보여 화재가 나지 않도 | 정보다."                     | 0.0964	False
하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 남북갈보리 성을  기자              | 0.0000	False
다. 대구 최세호 장병호 기자          | 다대 최세호 장병있다.              | 0.0000	False
의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 와의 동일 을 저이보이다.            | 0.0000	False
단송수관로 누수발생사고도 인재다. 구미시는 5월 16일 문서로 낙 | 귀국치을 면 환시리 이다.            | 0.0000	False
--------------------------------------------------------------------------------
[3200/100000] Train loss: 0.00462, Valid loss: 0.01067, Elapsed_time: 1232.64566
Current_accuracy : 3.500, Current_norm_ED  : 0.15
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
서 받은 자료를 바탕으로 책 두권 분량의 탄원서를 만들어 금감독 | 문 확까지 최세 호있다.             | 0.0000	False
정권자는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김 | 정보합서 열린0 리 두있다.           | 0.0000	False
검찰 관계자가 전했다. c검사는 "아들이 해외유학 때문에 고교 진학 | 입을로열지 환시를을 마 이 기자         | 0.0000	False
원 당시 합의를 이행하라"고 했다. 이 대표는 또 "5년 동안 연평균 30 | 와 총보위해니이 기정보습니다.          | 0.0000	False
문위원회'를 구성해 성 평등에 관한 전반적인 학교 정책을 논의할 예 | 남갈등을합 경찰이 이 정입니다.         | 0.0000	False
--------------------------------------------------------------------------------
[3300/100000] Train loss: 0.00141, Valid loss: 0.00483, Elapsed_time: 1268.28245
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
전입학시다는 일부 언론의 보도는사실무근이다"고 해명한 것으로 | 빛다. 격 고 있다.               | 0.0000	False
니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 성 환총를 을 경이 보 다.           | 0.0000	False
동 자택과 1억 원짜리 수표 30장 등 모두 58억 원 규몹니다. cj그룹 | 적의 동일에 등을니다."             | 0.0000	False
장을 고발했지만 무의 처리다. 당시 오 교사는 아는 검사를 통해 | 입 난 확최세다 것으 기자            | 0.0000	False
족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 단 총일해 해  있다.              | 0.0000	False
--------------------------------------------------------------------------------
[3400/100000] Train loss: 0.00079, Valid loss: 0.00723, Elapsed_time: 1303.61259
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
다. 이 관계자는 이어 "전입학 당시 오 교사가 전입학서류를 담당 교 | 공기보입 열다 리사직 기자            | 0.0000	False
이다. 소음이 많은 것이 단점으로 꼽히기는 하지만 전후 방향의 강력 | 남북갈등으면 산조당저 보이다.          | 0.0000	False
전혀 없었다"고 말했다. 앞서 프랑스 일간 리베라시은 로르 회장의 | 와바랜 일에 바 기자               | 0.0000	False
서는 등 사태가 갈수록 꼬여가고 있다. 한나라당은 전혀 다른 논리를 | 공 롯 열1y 0만원 달니다.          | 0.0000	False
로 삭감는데도 정파적으로 몰고가는 것은 아집과 독선"이라고 말했 | 날씨정보 보 기자                 | 0.0002	False
--------------------------------------------------------------------------------
[3500/100000] Train loss: 0.00075, Valid loss: 0.00634, Elapsed_time: 1339.29222
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
수 있게 하고, 고객들은 위생처리를 거친 우유를 안심하고 마실 수 있 | 을 환자일법에 촉각 곤두했다.          | 0.0000	False
다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 소해문 확대확대된 으보인다.           | 0.0000	False
과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 입을보입까지 산1이 보 기자           | 0.0000	False
이다.                       | 이다.                       | 0.4268	True
상태로 이뤄지는 콜드체인시스템을 도입해 2000여 곳의 전용목장에 | 성과는격 해입  기자               | 0.0000	False
--------------------------------------------------------------------------------
[3600/100000] Train loss: 0.00061, Valid loss: 0.00581, Elapsed_time: 1374.32438
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
민 1명이 다리에 총상을 입었다. 브라질 경찰은 300여명의 병력을 동 | 빛문이 정등을 면이 고 이다.          | 0.0000	False
법무부 공보관은 "김우중씨 명의로 된 한국 여권과 프랑스 여권을 모 | 적씨로 지전해드 있다.              | 0.0000	False
행복한 환경 속에서 자라야 한다는 생각은 '밀크마스터'(milk maste | 공북보해지 0을니다.               | 0.0000	False
대안으로 마련한 것으로 4대강사업을 반대해 온 김두관 지사에 대한 | 기북등격 이다.                  | 0.0000	False
부와 한은 간에 자료협조, 경제상황에 대한 의견교환 등 보다 긴밀한 | 날정보 기자                    | 0.1221	False
--------------------------------------------------------------------------------
[3700/100000] Train loss: 0.00062, Valid loss: 0.00530, Elapsed_time: 1410.01397
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
른 자료를 사용했기 때문으로 밝혀다. 감사원은 기재부장관에게 " | 공 롯호에 서에 0만직을곤 세다.        | 0.0000	False
감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 단다 총격 지최 정보습니다."          | 0.0000	False
고 거듭 밝다. 김형곤의 동생이 산 암태면 벌목도는 신안군내 753 | 남북갈등 산1이 사만보 기자           | 0.0000	False
14년까지 4년간 총 3600억 원을 지원할 계획이었다. 하지만 한나라 | 을랜지안 를 짐마 사이고 있다.         | 0.0000	False
고 더 큰 사회 문제를 일으키는 원인이  것"이라며 "결국 그들이 당 | 성의 영를 을 경이 기자             | 0.0000	False
--------------------------------------------------------------------------------
[3800/100000] Train loss: 0.00048, Valid loss: 0.00536, Elapsed_time: 1445.69787
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
불러다. 수자원공사는 지난 5월 사고 당시 다시는 사고가 발생하지 | 북갈보이다.                    | 0.0021	False
에 대해 구속영장을 청구했다. 검찰에 따르면 이씨는 2009년 10월 | 단과 총격을 니다 0이원 두다.         | 0.0000	False
사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 와 켜보일리에 촉을  기자            | 0.0000	False
사 사장은 연임이 유력한 것으로 알려다. 공기업 경영평가에서 a등 | 공 보입에린 당찰된 계획있다.          | 0.0000	False
사진을 위주로, 최근 일본 개봉 스줄이 나온 원 주연의 '우리형' | 빛과 구최 된 장병호 다자            | 0.0000	False
--------------------------------------------------------------------------------
[3900/100000] Train loss: 0.00040, Valid loss: 0.00454, Elapsed_time: 1480.82220
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
장을 고발했지만 무의 처리다. 당시 오 교사는 아는 검사를 통해 | 남 갈등입 이 기자                | 0.0000	False
어우러져 있는 그랑 마레농은 장어의 느끼한 맛을 잘 잡아준다. 또한 | 와 동보격 을니다 사망하이도 다.        | 0.0000	False
호사는 박 전 대통령의 현금 10억여 원도 보관 중입니다. 지금까지 드 | 빛씨정보  기자                  | 0.0020	False
법무부 공보관은 "김우중씨 명의로 된 한국 여권과 프랑스 여권을 모 | 1롯난호 지언 를드습 다.            | 0.0000	False
한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 딱북롯지 처1를 를앞 했다.           | 0.0000	False
--------------------------------------------------------------------------------
[4000/100000] Train loss: 0.00036, Valid loss: 0.00402, Elapsed_time: 1516.61946
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
간담회 직후 기자들과 만나 "앞으로 자주 만나기로 했고, 차관과 부총 | 한문 지확씨 다."                | 0.0000	False
적 코리코 303호(860t급)의 선원 7명 가운데 이날 오후 7시 40분께 | 성마 지환를 니1 0원니다."          | 0.0000	False
하기 위해 해외 직무파견 제도를 도입했다. 파견 대상은 대리급 근속 | 을 환보광법 원 기자               | 0.0000	False
우포 으 명소 가꾸기, 산청 한방휴양체험특화도시 조성, 합천 대 | 기북정보습  기자                 | 0.0264	False
의 일정 비율을 적립해 기부하게 된다. 차병원은 이기부금을 소아뇌 | 단과 총을 최 세다 것각보 다.         | 0.0000	False
--------------------------------------------------------------------------------
[4100/100000] Train loss: 0.00034, Valid loss: 0.00473, Elapsed_time: 1552.26670
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
"며 "600억원에 이르는 외자 유치를 한다고까지 부풀려져 황당하다" | 와 보갈등 y호 산짐 고 있다.         | 0.0000	False
규모로 설립된 원자력 서비스센터는실물 원자로와 연료장전 설비, 가 | 날북정보전 전해드니다.              | 0.0000	False
다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 1딱해지1 y저 사이도 했다.          | 0.0000	False
률상담이나 소송대리 등을 지원'하는 개정법률안을 8월 셋째 주에 발 | 빛동랜 광등 y 앞형 고 있다.         | 0.0000	False
성가족위원회 소속 이자스민 의원은 '일본군 위안부 피해자들에게 법 | 공 롯보지호에 촉각을 있다자           | 0.0000	False
--------------------------------------------------------------------------------
[4200/100000] Train loss: 0.00030, Valid loss: 0.00451, Elapsed_time: 1587.17808
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
여학생 중에 (성관계를) 하고 싶은 사람을 골라라는 말을 한 것으로 | 공 동법 일에 촉을 속인세다.          | 0.0000	False
다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 소문이 확 확된 것으종호 기자          | 0.0000	False
뒤 취한 조니다. 법원에 청구한 동결 대상 재산은 28억 원에 매입한 | 다.                        | 0.2271	False
금감독원이 금서비스개선국을 신설하고 외환업무실을 외환감독 | 적문 지텔리에 당한 기자             | 0.0000	False
4년형을 선고하는 것은 과도하다는 지적이다. 또 소 네트워크를 규 | 정으로 최된 것으호 기자             | 0.0000	False
--------------------------------------------------------------------------------
[4300/100000] Train loss: 0.00028, Valid loss: 0.00474, Elapsed_time: 1622.90123
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
치내가 산 것으로 오해하고 있다"고 말했다. 그는 "동생이 섬을 사기 | 빛 켜보입 니 사형앞 원바니다.         | 0.0000	False
로 소비자의 변화를 반영한 개발 및 마팅에 열심이다. | 성마 환를 이 원이다.              | 0.0000	False
임원급 인사와 함께 발표할 예정이다. 개편안에 따르면 기존 감독서 | 소문이 영광입니y 시기를 원 기.        | 0.0000	False
발견, 경찰에 신고 했다. 이씨는 경찰에서 "집에 돌아와 보니 함께 지 | 기 갈보등보 열찰이 사원직호 다.        | 0.0000	False
수 있는 구조로 설계되어 있다. 이렇게 되면 견인성과 제동성이 클 뿐 | 남북등보 입다 경찰이 보있다.          | 0.0000	False
--------------------------------------------------------------------------------
[4400/100000] Train loss: 0.00027, Valid loss: 0.00453, Elapsed_time: 1658.67827
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
고 인터넷 카페에 글을 올려 돈을 받고 팔다"고 말했다. | 성딱환 특등을 확저할 있다.           | 0.0000	False
생활고에 시달리다 남편과 동반자살을 하기로 한 뒤 혼자남는 딸이 | 공동 랜광 1찰 이길 보 다."         | 0.0000	False
로 고용했으며 로르 회장은 2003년 이래 김 전회장을 최소 세 번 만 | 남북정등으로 산찰이 보 니다.          | 0.0000	False
자들의 명예가 손 우려가 있는 사건들이 늘고 있다"며 "일본군 위 | 소문이 조짐된 장앞로한 기자           | 0.0000	False
"나흘 간의 설 연휴가 시작습니다. 연휴 첫 날, 잘 보내고 계신가요? | 공 롯데호에 산당을 길 고계 있다.       | 0.0000	False
--------------------------------------------------------------------------------
[4500/100000] Train loss: 0.00024, Valid loss: 0.00440, Elapsed_time: 1693.85742
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
는데 회동 장소 중 한 곳이 서울이었다고 전했다. 1999년 10월 출 | 남정보습 다.                   | 0.0002	False
국정원 뇌물 사건을 맡은 유영하 변호사가 보관하고 있습니다. 유 변 | 남 총등을 벌인저 사이이다."          | 0.0000	False
법을 "최악의 사태를 몰아오는 정치적 도발"로 규정했다. 북한은 논평 | 빛바랜 영격을 1n경길 획있다.         | 0.0000	False
9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 남북갈등보까 확1 박촉정보니다."        | 0.0000	False
고생할 것으로 우려해 딸에게 극약을 먹여 숨지게 한 의로 구속기 | 북정보산 을 저 보고호있기자           | 0.0000	False
--------------------------------------------------------------------------------
[4600/100000] Train loss: 0.00026, Valid loss: 0.00464, Elapsed_time: 1729.72074
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.800, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
생활고에 시달리다 남편과 동반자살을 하기로 한 뒤 혼자남는 딸이 | 공동 랜광 서찰 이길 보 니다."        | 0.0000	False
인 피고인이 잘못을 뉘우치며 정신적 후유증에 시달리고 있는 점 등 | 빛바랜 영광위해지 을 다.            | 0.0000	False
밝다. 박씨도 로비 자체를 부인하고 있다. 박씨는 부산저축은행에 | 한날씨보 해보습 기자               | 0.0000	False
다. 조 전 수석은 박 전 대통령이 이 부회장 퇴진을 지시했고 손경식 c | 빛 랜보 산린이 사촉직병호 기자         | 0.0000	False
"문재인 대통령이 남북 문제와 관련해 강력한 국방력을 기반으로 대 | 와 영일 세 것으로보다.             | 0.0000	False
--------------------------------------------------------------------------------
[4700/100000] Train loss: 0.00021, Valid loss: 0.00395, Elapsed_time: 1764.78500
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
을 통해 "북한의 협박에 굴복해 법제정을 망설인다면 국회가 본분을 | 공 롯일 서세 저보호 기자            | 0.0000	False
에 입맛이 떨어지기 십상인 여름에 더할 나위 없이 좋다. 초복을 시작 | 남북갈보법안 리 했다.              | 0.0000	False
입을 비롯한 다양한 의혹이제기에 따라 이달 20일부터 c검사에 대 | 입을까지면산서 저고있다.             | 0.0000	False
0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 입을열지 데일리 속세다.             | 0.0000	False
있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 1 총격위지이 사만을 곤두 다.자        | 0.0000	False
--------------------------------------------------------------------------------
[4800/100000] Train loss: 0.00021, Valid loss: 0.00594, Elapsed_time: 1801.25448
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
여학생 중에 (성관계를) 하고 싶은 사람을 골라라는 말을 한 것으로 | 공 호 일 박것를 습속다."           | 0.0000	False
된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 정해다.                      | 0.1942	False
여야 원내대표간 합의했던 등록금 부담완화 문제와 저축은행 국정조 | 딱보일 전해드보밝다"               | 0.0000	False
로 보여 개막식 행사에 참여하는 분들은 추위에 대비를 잘 해주야 | 한 지해지는 등을 구 있다.           | 0.0000	False
률상담이나 소송대리 등을 지원'하는 개정법률안을 8월 셋째 주에 발 | 빛바랜 영광등 난 를앞원길 바 했다.      | 0.0000	False
--------------------------------------------------------------------------------
[4900/100000] Train loss: 0.00021, Valid loss: 0.00474, Elapsed_time: 1836.86997
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
놔두기 어렵다는 점과 3억원이 조씨에게 전달다는 여러 증거가 드 | 단 총격을 세1다 0원곤호 기자         | 0.0000	False
생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 딱딱일 을앞 기자                 | 0.0000	False
몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 입을보다.                     | 0.0031	False
적 높게 일습니다. 기상정보습니다."      | 적 높게 일습니다. 기상정보습니다."      | 0.1030	True
입을 비롯한 다양한 의혹이제기에 따라 이달 20일부터 c검사에 대 | 입을까지 산서 저있다.              | 0.0000	False
--------------------------------------------------------------------------------
[5000/100000] Train loss: 0.00018, Valid loss: 0.00493, Elapsed_time: 1872.10765
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
건강하고 품위있는 노년을 위한 정책을 만들다고 말했습니다. 어르 | 북보해지1 앞을  있다.             | 0.0000	False
사용할 계획이다. 이재우 사장은 "이번 협약 체결로 병마와 싸우고 | 입을이 전다.                   | 0.0005	False
교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 적 높보일해 해했다.               | 0.0000	False
위반)로 김모(18)군을 불구속 입건했다. 경찰에 따르면 김군은 고교 | 을 지켜호에 을서린 0만직오종호 기자      | 0.0000	False
효중인 가운데 물결이 거세게 일습니다. 내일 아침에는 서울의 기 | 공 총보격해열면 열다. 사형원습니다.      | 0.0000	False
--------------------------------------------------------------------------------
[5100/100000] Train loss: 0.00018, Valid loss: 0.00465, Elapsed_time: 1907.84689
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
재 선에서 협의채널을 만들기로 했다"고 말했다. 양측은 간담회 직후 | 성마 여를  기자                 | 0.0001	False
의할 예정이라고 8일 밝다. 이 의원이 밝 개정안 내용은 '일제하 | 을 지롯며입 열린다 데찰린 직오고 있다.    | 0.0000	False
후배가 되는 것을 싫어하는 것을 보고 아내의 지인이 사는 강동구로 | 한다" 해 기자                  | 0.0149	False
부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 단다 총격해지 확조으을갖당길 계획이다.     | 0.0000	False
로 소비자의 변화를 반영한 개발 및 마팅에 열심이다. | 성마비환를 이 이원이다.             | 0.0000	False
--------------------------------------------------------------------------------
[5200/100000] Train loss: 0.00018, Valid loss: 0.00461, Elapsed_time: 1943.56262
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
제하는 것은 표현의 자유를 가로막을 뿐이며 정부가 반정부 시위 때 | 와 일여 등을 확저 예정이다.          | 0.0000	False
연합뉴스와의 전화 통화를 통해 "최근 변호사인 친동생(김형진 씨)이 | 심을 확산을갖하 저이다.             | 0.0000	False
로 승승장구했던 대표팀,최종 예선전 성적표는 참혹했습니다.지난해 | 남 동일안 등을 사원 다.            | 0.0000	False
사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 공북갈보호서면 환서 데방에 직종 기자      | 0.0000	False
육관광부 우수축제로 선정된 '제81회 춘향제'가 열린다. 정시에서 | 심문는치취지 조리 를 있다.           | 0.0000	False
--------------------------------------------------------------------------------
[5300/100000] Train loss: 0.00017, Valid loss: 0.00519, Elapsed_time: 1978.56675
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
다.                        | 다.                        | 0.9071	True
이 각각 생보검사국와 손보검사국으로 바뀌고 보험감독국이 분리된 | 기 동보등 습저 보니다."            | 0.0000	False
영문 제목)'는 한국에서 1천100만명을 모은 '브라더후드('태극기 휘 | 공 켜보해텔에 0만을 보있다.          | 0.0000	False
별법 적용을 받게 니다. 조원동 전 청와대 경제수석에 대한 재판은 | 남상갈정보습니 확저 보정있다.          | 0.0000	False
타깝다며 인성 교육을 더욱 강화하다고 밝다.학교 측은 이번 성 | 성마랜 영격구 최세 장으보두다.         | 0.0000	False
--------------------------------------------------------------------------------
[5400/100000] Train loss: 0.00017, Valid loss: 0.00587, Elapsed_time: 2014.15261
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
건강하고 품위있는 노년을 위한 정책을 만들다고 말했습니다. 어르 | 딱보해니다."                   | 0.0000	False
주민의 인권을 실질적으로 증진하고 국제적 기준에 따라 인도적 지원 | 남정보 기자                    | 0.0479	False
다. 문 대통령은 오늘 대한노인회 회장단과 가진 오찬에서 과거처럼 | 남 갈등을 벌인 경찰이 다.           | 0.0000	False
금실무협의회 이후 십수년만에 부활되는 것이다. 하지만 재정부가 | 입을열 지시기 를앞을 고 있다.         | 0.0000	False
비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 와 동일 니당저 고 있다.            | 0.0000	False
--------------------------------------------------------------------------------
[5500/100000] Train loss: 0.00015, Valid loss: 0.00530, Elapsed_time: 2049.84572
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
고 인터넷 카페에 글을 올려 돈을 받고 팔다"고 말했다. | 성마지환 를등을 확저할 있다.          | 0.0000	False
감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 정보다.                      | 0.1519	False
기 정보를 확인하야습니다. 오늘 낮 최고 기온은 서울 2도 등 어 | 남0을해1면 0조리 앞 기자           | 0.0000	False
시한 결과다. 경북도와 구미시는 물론 가물막이 아래 준설공사를 담 | 정보산등을확인저할 예정이다.           | 0.0000	False
말을 인용, 로르 회장이 한국에서사업을 위해 김 전 회장을 고문역으 | 공북갈등으로지 면 조직 오종호 기자       | 0.0000	False
--------------------------------------------------------------------------------
[5600/100000] Train loss: 0.00016, Valid loss: 0.00422, Elapsed_time: 2084.90976
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 딱해일 을앞 기자                 | 0.0000	False
은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 성바 보일 있다.                 | 0.0002	False
있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 1과 총격위지이 사만을 두 다.자        | 0.0000	False
무를 넘어서는 것임을 우회적으로 비판하며, 압박한 것이다. 최 장관 | 소문랜이 확광을짐 경저 망이 기자        | 0.0000	False
오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 와 부 을다.                   | 0.0001	False
--------------------------------------------------------------------------------
[5700/100000] Train loss: 0.00013, Valid loss: 0.00445, Elapsed_time: 2121.35130
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한국노총도 법안 강행시 벌이기로 했던 노사정위 탈퇴와 대정부 투쟁 | 남북등으로까지 확산조짐 저이보 기자       | 0.0000	False
의로 기소된 노모(38.여)씨에 대해 징역 3년에 집행유예 5년을 선 | 날씨으보 지 0앞사 원곤호 기자         | 0.0000	False
인하대는 의예과 남학생들의 여학생 집단 성희과 관련해 9일 공식 | 소문지환를 을y저 원했다.            | 0.0000	False
무를 넘어서는 것임을 우회적으로 비판하며, 압박한 것이다. 최 장관 | 소문이 확등 합짐마 저원이다.          | 0.0000	False
다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 소문이 확 확세된 것으종호 기자         | 0.0000	False
--------------------------------------------------------------------------------
[5800/100000] Train loss: 0.00014, Valid loss: 0.00474, Elapsed_time: 2156.39481
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 성 일 해을니다.                 | 0.0010	False
검찰에 비공식적으로 자진귀국 의사를 타진한 적이있었으나 같은해 | 날씨정보다.                    | 0.0494	False
공동브리핑에서 "우리 경제가 수출 호조와 고용 개선 등에 힘입어 잠 | 입을열면 소시리를앞 보두있다.          | 0.0000	False
으로 산뜻하고도 정교한 맛이 부드럽고 감칠맛 있는 민어회와의 조화 | 남북정보 산y 저 보니다.            | 0.0000	False
공조를 강화하기 위해서다. 박재완 기획재정부 장관과 김중수 한국은 | 입을 열면 면서 린 사망직호 기자        | 0.0000	False
--------------------------------------------------------------------------------
[5900/100000] Train loss: 0.00013, Valid loss: 0.00409, Elapsed_time: 2192.08581
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
비 부담을 덜어주기 위해 주40시간제를앞당겨 시행하면서 정규직을 | 1 보 지호 를당 기자              | 0.0000	False
7호 객차에 앉아 졸고 있었는데 갑자기 주위가 시끄러워 눈을 떠 보 | 날씨정보  기자                  | 0.0019	False
제 상황처럼 해볼 수 있게 돼정비 전문인력 양성은 물론, 새로운 정비 | 기북정보습 박전보습니다."            | 0.0000	False
개월 만에 불명예 퇴진했습니다.본선행에 빨간 불이 켜진 대표팀을 | 남북갈등보까 산 조짐마이 사하기고 있다.    | 0.0000	False
원의 중간 전달자로 지목된 조기문씨가 돈을 받은 3월 15일과 다음날 | 빛씨 북보일보 전으로 보니다.          | 0.0000	False
--------------------------------------------------------------------------------
[6000/100000] Train loss: 0.00013, Valid loss: 0.00381, Elapsed_time: 2227.69354
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
글을 올다는 의로 기소된조던 블쇼(20)와 페리 서트클리프 키 | 남정보습까 산조짐마저 이보 기다.        | 0.0000	False
에 참석한 민주노총 한 핵심관계자는 "참여정부에서 가장 실패한 정 | 남북갈등보까확산 앞저 있기자           | 0.0000	False
니다. 날이 풀리면서 내일 오전에 미세먼지 농도가 높아질 수 있어 주 | 공 롯보호에 소시린 이이다.           | 0.0000	False
면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 남북갈보로 지세 장보있다.            | 0.0000	False
고 있습니다. 화재 예방에 유의하야습니다. 내일은 전국에 구름 | 남해보산입 앞고 기자               | 0.0000	False
--------------------------------------------------------------------------------
[6100/100000] Train loss: 0.00012, Valid loss: 0.00404, Elapsed_time: 2262.63118
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
권 수익 전액을 일본 유니세프에 기부하기로 했다. 원의 매니저 장 | 단과 총격을 입니면 조원당 고 있다.      | 0.0000	False
의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 날씨정보해 전 전보있다.             | 0.0000	False
밝다. 박씨도 로비 자체를 부인하고 있다. 박씨는 부산저축은행에 | 날보 기자                     | 0.0022	False
감독국으로 확대된다. 또 외은지점의 감독과 검사를 전담하는 별도의 | 한는 해지 21세린 박각곤보두다.        | 0.0000	False
황토현 동학축제'를 통해 1박2일 황토현숙영캠프, 동학농민명군 | 단 총격을 1이 조길 획했다.          | 0.0000	False
--------------------------------------------------------------------------------
[6200/100000] Train loss: 0.00013, Valid loss: 0.00422, Elapsed_time: 2298.41280
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
습니다. 문 대통령은 또 평창 올림픽을 평화 올림픽으로 만들어, 북핵 | 와의 동보일호 를언를 한 보세다.        | 0.0000	False
인 16일 휴대전화 위치 추적을 벌인 결과 현 전 의원과 동선이 일치하 | 딱보지 산당세y 보이원고 있다.         | 0.0000	False
인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 단과 총보등 니 원전 있다.           | 0.0000	False
적 코리코 303호(860t급)의 선원 7명 가운데 이날 오후 7시 40분께 | 성 지환를 을1 0만원 니다."         | 0.0000	False
과 유럽 재정위기 등 불안요인이 크고 대내적으로는 물가불안, 가계 | 적 환 을니 기전를보습니다."          | 0.0000	False
--------------------------------------------------------------------------------
[6300/100000] Train loss: 0.00012, Valid loss: 0.00411, Elapsed_time: 2334.12843
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 기상정보습 전습니다."              | 0.0027	False
건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 소로 전를  계획있다.              | 0.0000	False
결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 성마 영일촉 를드병호 기자            | 0.0000	False
들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 10롯데호 열에린 앞저 보호 기자        | 0.0000	False
유효슈팅 없이 0대 1로 완패한 데 이어, 6차전에서 다시 만난 중국에 | 기 지북보 찰이사 기자              | 0.0000	False
--------------------------------------------------------------------------------
[6400/100000] Train loss: 0.00011, Valid loss: 0.00419, Elapsed_time: 2369.13064
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
무를 넘어서는 것임을 우회적으로 비판하며, 압박한 것이다. 최 장관 | 소문랜이 확광을 경저 사망이 기자        | 0.0000	False
3건 25억7000여만원을 삭감했다. 여기에는 김 지사의 대표적인 공 | 성문 를 을 경저이고 했다.           | 0.0000	False
공음면에서는 지난 23일부터 '청보리 축제'가 한창이다. 올해 8회를 | 북정보 전해드 기자                | 0.0000	False
수 있는 구조로 설계되어 있다. 이렇게 되면 견인성과 제동성이 클 뿐 | 단는 치 치입면 조찰저 보이다.         | 0.0000	False
시한 결과다. 경북도와 구미시는 물론 가물막이 아래 준설공사를 담 | 정보산등을 확인저할 예정이다.          | 0.0000	False
--------------------------------------------------------------------------------
[6500/100000] Train loss: 0.00011, Valid loss: 0.00380, Elapsed_time: 2404.77337
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
금실무협의회 이후 십수년만에 부활되는 것이다. 하지만 재정부가 | 공동 롯광을 1 를앞 기자            | 0.0000	False
불 축제'를 연다.                | 불 축제'를 연다.                | 0.3310	True
목할 것으로 검찰은 보고 있다. 검찰은 현 전 의원에 대한 전방위 자금 | 기정보습이다.                   | 0.0092	False
. 타이어는 재질 역시 스노우 타이어의 제동력을 높이는 역할을 하게 | 공 켜보호 1기 앞각을 곤 있다.        | 0.0000	False
찾는 이들이 많다. 하지만 여름 보양식에 와인을 곁들이면 더욱 좋다 | 딱 해법1 앞 기자                | 0.0000	False
--------------------------------------------------------------------------------
[6600/100000] Train loss: 0.00012, Valid loss: 0.00386, Elapsed_time: 2440.46346
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
체 호워드 리그의 드루 닐슨은 "징역 4년형은 흉기로 상해를 입 | 심로 를으를 기자                 | 0.0000	False
안부 피해자들이 보다 적극적으로 자신의 명예와 존엄성을 지킬 수 | 와 동일여 등을 사보이고 했다.         | 0.0000	False
는데 회동 장소 중 한 곳이 서울이었다고 전했다. 1999년 10월 출 | 정보 다.                     | 0.1000	False
정보습니다."                   | 정보습니다."                   | 0.4627	True
이다. 소음이 많은 것이 단점으로 꼽히기는 하지만 전후 방향의 강력 | 남북갈등으로까 산조당저 보 다.         | 0.0000	False
--------------------------------------------------------------------------------
[6700/100000] Train loss: 0.00011, Valid loss: 0.00424, Elapsed_time: 2475.59888
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
인하대는 의예과 남학생들의 여학생 집단 성희과 관련해 9일 공식 | 소문소지환를 을당y저 앞원했다.         | 0.0000	False
지만 아들의 답안지 작성에 관여한 일이 없다"며 결백을 주장한 것으 | 입을합면 서열린다 사직오보 다.         | 0.0000	False
는 이 자리에서 "민주노총은 운명적으로 정권교체를 함께해야 할 파 | 을 지환안 를을니다 사앞원입니다.        | 0.0000	False
검찰 관계자가 전했다. c검사는 "아들이 해외유학 때문에 고교 진학 | 북갈보환 산를등을 망하호 기자          | 0.0000	False
두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 소문이확한이다.                  | 0.0002	False
--------------------------------------------------------------------------------
[6800/100000] Train loss: 0.00010, Valid loss: 0.00399, Elapsed_time: 2511.25155
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
서울우유가 치열한 유업체들간 경쟁시장에 '행복' 발을 들고 나 | 소문 확로 지 정보습니다."           | 0.0000	False
대북 정책을 믿고 국론을 하나로 모아달라고 당부했습니다. kbs 뉴스 | 날정원으  있다.                 | 0.0000	False
교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 적 지보며를일에 를  있다.           | 0.0000	False
 등 미네 성분과 비타민이 풍부해 여름철에 부족할 수 있는 미네 | 남 총등을해난산 박이정다."           | 0.0000	False
록 조심하야습니다. 아침 기온은 서울 영하 5도, 대구는 영하 3도 | 을 켜보며광에다 y 것직 보호다"        | 0.0000	False
--------------------------------------------------------------------------------
[6900/100000] Train loss: 0.00010, Valid loss: 0.00474, Elapsed_time: 2546.40461
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
서울시 교육청 관계자 등은 20일 c군이 b고에 전입학한 작년 3월 가 | 입을까입 조짐고 이다.              | 0.0000	False
에 참석한 민주노총 한 핵심관계자는 "참여정부에서 가장 실패한 정 | 남북등보까확산 앞저 있기자            | 0.0000	False
는 23일 법안심사소위를 열어 정부가 제출한 비정규직법안을 심사해 | 빛 랜보광입니안 y저 형원입니이다.       | 0.0000	False
박 전 대통령은 재산 몰수와 추징 시효가 10년으로 늘어난 전두환 특 | 딱 켜환광 광니1y 0원 니다.         | 0.0000	False
다. 조 전 수석은 박 전 대통령이 이 부회장 퇴진을 지시했고 손경식 c | 기북보 산린이 사촉직을 호 기자         | 0.0000	False
--------------------------------------------------------------------------------
[7000/100000] Train loss: 0.00010, Valid loss: 0.00438, Elapsed_time: 2581.95203
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
영국 폭동 와중에 페이스북에 선동 글을 올린 젊은이들에게 중형이 | 딱해지 발조을 바 있다.             | 0.0000	False
각종 모의시험 설비를 갖춘 원자력 서비스센터를 준공했다. 원자력 | 적날비 게를  드습니다."            | 0.0000	False
정보습니다."                   | 정보습니다."                   | 0.3546	True
놔두기 어렵다는 점과 3억원이 조씨에게 전달다는 여러 증거가 드 | 단 격을 세호 0원곤호 기자           | 0.0000	False
다. 서울우유는 지난 11일 75주년 기념식에서 '행복한 젖소, 행복한 | 1 기자                      | 0.1828	False
--------------------------------------------------------------------------------
[7100/100000] Train loss: 0.00011, Valid loss: 0.00426, Elapsed_time: 2617.68661
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 공 롯데호법안 방에 앞저리곤두 기자       | 0.0000	False
부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 단0총격해입 전를앞고  했다.          | 0.0000	False
이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 남0조을합지면소 소전 니다.           | 0.0000	False
제협력국이 기획조정국, 총무국, 공보실과 함께 기획총 부문으로 | 소문 지일 를로 당세호 전형습다.        | 0.0000	False
기로 했다.남학생들이 징계무효확인 소송을 다는 사실이 알려지자 | 공 켜보호 처만언 을 한 있다.         | 0.0000	False
--------------------------------------------------------------------------------
[7200/100000] Train loss: 0.00009, Valid loss: 0.00443, Elapsed_time: 2652.68097
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
맞는 이 축제는 공음면 학원농장의 보리밭에서 5월 8일까지 열린다. | 단과 격위광 짐저 보이고 기자          | 0.0000	False
의 일본식 명칭)는 일본 고유의 영토'라고 적은 말을 묶고, 매춘부 | 의문 등을 보 기자                | 0.0000	False
과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 입을열면 환1저 사촉오종호 기자         | 0.0000	False
우포 으 명소 가꾸기, 산청 한방휴양체험특화도시 조성, 합천 대 | 기상정보습  기자                 | 0.0402	False
대할 계획이다.                  | 대할 계획이다.                  | 0.0885	True
--------------------------------------------------------------------------------
[7300/100000] Train loss: 0.00010, Valid loss: 0.00443, Elapsed_time: 2688.28761
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 소 지격구 최세 사앞원 있다.          | 0.0000	False
독립 투쟁을 위해 활동한 장소를 둘러본다. 또 고구려 유적지, 광개토 | 딱 롯해 드니다."                | 0.0001	False
온 영하 6도까지 떨어지는 등 반짝 추위가 이어지습니다. 하지만, | 남갈보여 등을합면 사원정 했다.         | 0.0000	False
두확인한 결과, 김우중 전 회장이 출국한 이후 어느 쪽 여권으로도 귀 | 날보 박씨를 습니다."              | 0.0000	False
만한 보양식이 없다. 칼칼한 낙지볶음에는 '크로포드 소비 블랑' | 입보켜지법 일 박언각을 앞두호 기.자      | 0.0000	False
--------------------------------------------------------------------------------
[7400/100000] Train loss: 0.00009, Valid loss: 0.00451, Elapsed_time: 2724.01560
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
여학생 중에 (성관계를) 하고 싶은 사람을 골라라는 말을 한 것으로 | 적 동 방에 언각을 다.             | 0.0000	False
장을 향해 또다시 목소리를 높다. 최 장관은 30일 기자간담회에서 | 빛 열환 을당길 계이 했다.           | 0.0000	False
인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 10조을해1면 0조원을 고 있다.        | 0.0000	False
0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 입을 지에 조방에 을 기다.           | 0.0000	False
법무부는 21일 담임교사가 아들의 답안지를 대신 작성한 사건으로 | 공동 롯데호텔에 린짐 찰저 계있다.       | 0.0000	False
--------------------------------------------------------------------------------
[7500/100000] Train loss: 0.00008, Valid loss: 0.00491, Elapsed_time: 2758.97699
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
률상담이나 소송대리 등을 지원'하는 개정법률안을 8월 셋째 주에 발 | 빛바랜 광 산당 앞당길 고 있다.        | 0.0000	False
다.                        | 다.                        | 0.8480	True
과 축제 주점 등지에서 같은 과 여학생들을 언급하며 성희을 했고, | 한 롯해지는 발 를을 한 있다.         | 0.0000	False
계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 을 랜롯광입니y 0만앞 오호 기자        | 0.0000	False
조되는 과정에서 높은 파도로 많은 물을마 의식불명 상태로 삼척의 | 빛 환일 등을 확인 원하고 다.         | 0.0000	False
--------------------------------------------------------------------------------
[7600/100000] Train loss: 0.00008, Valid loss: 0.00411, Elapsed_time: 2794.88970
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
국정원 뇌물 사건을 맡은 유영하 변호사가 보관하고 있습니다. 유 변 | 남 여등을 확저 정원이다."           | 0.0000	False
9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 남북갈보까 확1 박촉각오보호니다."       | 0.0000	False
심야까지 회의를 열지 못하고 24일로 연기했다. 법안심사소위 위원 | 한과 광 입된저 다.               | 0.0000	False
각 중단하고 박 원내대표에게 의가 있다면 당당히 기소하라"며 "한 | 기상보습입열세다 사형고 기자           | 0.0000	False
창 올림픽은 88올림픽 이후 성장과 발전을 전세계에 알리는 좋은 계 | 소문이 광1 앞한 다자              | 0.0000	False
--------------------------------------------------------------------------------
[7700/100000] Train loss: 0.00009, Valid loss: 0.00417, Elapsed_time: 2830.69315
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
귀국치 않다.                   | 귀국치 않다.                   | 0.2777	True
면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 와 격 최세 사한 바 있다.           | 0.0000	False
4년차 이상 실무자로 6개월 간 현지 문화와 언어를 익히고 시장 조사 | 단과 총격을보호 소시이 보병호 기자       | 0.0000	False
있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 1 롯을위지이 박씨각을 다.           | 0.0000	False
안부 피해자들이 보다 적극적으로 자신의 명예와 존엄성을 지킬 수 | 와 동일여 를 기자                | 0.0000	False
--------------------------------------------------------------------------------
[7800/100000] Train loss: 0.00008, Valid loss: 0.00464, Elapsed_time: 2865.75658
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
블록패을 띄고 있다. 자동차가 눈길을 주행할 때 바퀴가 눈을 누르 | 남북갈으로 전조앞로앞저보 기자          | 0.0000	False
어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 |  지켜일안 을  기자               | 0.0000	False
가 훌하다.                    | 가 훌제개하다.                  | 0.0322	False
식이 열리는 저녁에는 체감온도가 영하 10도 안팎으로 떨어질 것으 | 기정보습 전 있기자                | 0.0002	False
독립 투쟁을 위해 활동한 장소를 둘러본다. 또 고구려 유적지, 광개토 | 딱 롯해 드니다."                | 0.0001	False
--------------------------------------------------------------------------------
[7900/100000] Train loss: 0.00007, Valid loss: 0.00440, Elapsed_time: 2901.50964
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
경영실적보고서를 평가편람 내용과 다르게 작성하지 말라"고 주의를 | 남과랜격을 찰다. y이사망기 기자        | 0.0000	False
과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 입등으로까 산짐저 보이고 기자          | 0.0000	False
검찰에 비공식적으로 자진귀국 의사를 타진한 적이있었으나 같은해 | 날씨정보다."                   | 0.1394	False
맞는 이 축제는 공음면 학원농장의 보리밭에서 5월 8일까지 열린다. | 단다 격구 최세 저 망고 있다.         | 0.0001	False
건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 소을로 확를 당하저 계획 다.          | 0.0000	False
--------------------------------------------------------------------------------
[8000/100000] Train loss: 0.00008, Valid loss: 0.00447, Elapsed_time: 2936.51993
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
니다. 앞으로 대기는 점점 더 건조해 질 것으로 보여 화재가 나지 않도 | 북해입면 소를보니다."              | 0.0000	False
한인권법을 놓고 하게 맞서는 동안 북한은 이같은 움직임에 대한 | 남북해보 산를앞을한 바있다.           | 0.0000	False
위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 빛바 일 일습 전드습니다."           | 0.0000	False
선고다. 글랜드 체스터 지방법원이 페이스북에 폭동을 선동하는 | 빛다 북등으보 환찰0 사촉각을 기자       | 0.0000	False
채용할 경우 증원인력 1인당 월 50만원(올해 60만원)씩한시적으로 | 남정보습니 산이고 있다.             | 0.0000	False
--------------------------------------------------------------------------------
[8100/100000] Train loss: 0.00007, Valid loss: 0.00427, Elapsed_time: 2972.24566
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. |  일 를앞당를 구 인다.             | 0.0000	False
러나면서 현 전 의원이 종착지인지 여부를 밝히는 데 조씨의 진술이 | 을 지켜환며법리안 촉을  기자          | 0.0000	False
대구지법 제 12형사부(재판장 김종필 부장판사)는 1일 생활고로 남 | 을 환일부  기자                 | 0.0000	False
사 등 다른 사안과 연계 가능성까지 언급해 민주당의 반발을 고, 북 | 입을보열입 조찰이 사리이다.           | 0.0000	False
나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 빛과 랜광위입 경찰이 기자            | 0.0000	False
--------------------------------------------------------------------------------
[8200/100000] Train loss: 0.00008, Valid loss: 0.00445, Elapsed_time: 3007.81573
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 공딱북보1는 를 을한 다.            | 0.0000	False
청구했다. 잦은 말바꾸기로 증거인멸 등의 우려가 있는 조씨를 그냥 | 와의 일부 언등 당길 기자            | 0.0000	False
으며 3억원의 최종 종착지가 현 전 의원이라는 진술도 신하고 있다. | 성과 랜총 해10만원을 호 했다.        | 0.0000	False
대안으로 마련한 것으로 4대강사업을 반대해 온 김두관 지사에 대한 | 대딱피해보  이다.                | 0.0000	False
리고 모레 밤에는 경기 북부와 강원 영서 북부에 비가 내리습니다. | 한는지해지 산짐 저고 있다.           | 0.0000	False
--------------------------------------------------------------------------------
[8300/100000] Train loss: 0.00007, Valid loss: 0.00446, Elapsed_time: 3042.76254
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
로 삭감는데도 정파적으로 몰고가는 것은 아집과 독선"이라고 말했 | 기정보습 기자                   | 0.0083	False
에 입맛이 떨어지기 십상인 여름에 더할 나위 없이 좋다. 초복을 시작 | 남북보법 리했다.                 | 0.0001	False
주민의 민생 문제도 그대로 반영된다"고 덧붙다. 이처럼 여야가 북 | 남정보습 전이 소사이 고 있다자         | 0.0000	False
양구경찰서에 육군 모 사단에서 복무중인 사병 이모(22)씨가 112전 | 남북갈보지y산 앞이다.              | 0.0000	False
감독국으로 확대된다. 또 외은지점의 감독과 검사를 전담하는 별도의 | 한는 해지최호 1세린기박각곤보두다.       | 0.0000	False
--------------------------------------------------------------------------------
[8400/100000] Train loss: 0.00007, Valid loss: 0.00425, Elapsed_time: 3078.18120
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
검찰이 박근혜 전 대통령에 대한 재산 동결 조치를 취했습니다. 내곡 | 입을 열 소열시린 사리을 있다.         | 0.0000	False
몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 을 지켜며법안 향방에 촉각을 곤 있다.     | 0.0020	False
개국 8개 법인에 파견하기로 했다. 이외에도 대한통운은 다양한 사내 | 공북지보 처1일 언을 곤두했다.         | 0.0000	False
로 삭감는데도 정파적으로 몰고가는 것은 아집과 독선"이라고 말했 | 날정보 보습 기자                 | 0.0031	False
무를 넘어서는 것임을 우회적으로 비판하며, 압박한 것이다. 최 장관 | 소문이 광으을 세저 경망이했다.         | 0.0000	False
--------------------------------------------------------------------------------
[8500/100000] Train loss: 0.00008, Valid loss: 0.00485, Elapsed_time: 3113.70219
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
고 소개했다.                   | 고 소개했다.                   | 0.8023	True
무를 넘어서는 것임을 우회적으로 비판하며, 압박한 것이다. 최 장관 | 소문이 확광짐 경저 사망이 기자         | 0.0000	False
차량에 많이 쓰인다. 반면 래그형 은 트럭이나 버스에 장착되는 타이 | 남씨정보 제 획 있다.              | 0.0000	False
권 수익 전액을 일본 유니세프에 기부하기로 했다. 원의 매니저 장 | 단과 격을 입면 조앞원당 있다.         | 0.0000	False
로 정하고 추진했다. 이들 사업에 도는 사업당 200억 원을 한도로 20 | 1 보만향에  기자                | 0.0000	False
--------------------------------------------------------------------------------
[8600/100000] Train loss: 0.00007, Valid loss: 0.00555, Elapsed_time: 3148.71144
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 을 지켜며법안 향방에 촉각을 곤 있다.     | 0.0014	False
있었을 것이라면서 김 전 회장의 입국설에 회의적인반응을 보다. | 입 열다 환시리 앞바 있다.           | 0.0000	False
펴고 있다. 배은희 대변인은 16일 논평을 통해 "북한인권법은 북한 | 와 동일광위1 발를 한 기자           | 0.0000	False
청구했다. 잦은 말바꾸기로 증거인멸 등의 우려가 있는 조씨를 그냥 | 와의 일부 산당 당길 기자            | 0.0000	False
전혀 없었다"고 말했다. 앞서 프랑스 일간 리베라시은 로르 회장의 | 입을보 안광 를짐 앞 기자            | 0.0000	False
--------------------------------------------------------------------------------
[8700/100000] Train loss: 0.00007, Valid loss: 0.00405, Elapsed_time: 3184.36643
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 1 롯일호 를 당된으 보인다.          | 0.0000	False
정보습니다."                   | 정보습니다."                   | 0.3641	True
받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 을문지시리 을 있다.               | 0.0003	False
과 축제 주점 등지에서 같은 과 여학생들을 언급하며 성희을 했고, | 한 롯해지는 발를을 한 있다.          | 0.0000	False
고 인터넷 카페에 글을 올려 돈을 받고 팔다"고 말했다. | 성마지환 를특을 갖 고저할고 있다.       | 0.0000	False
--------------------------------------------------------------------------------
[8800/100000] Train loss: 0.00007, Valid loss: 0.00518, Elapsed_time: 3220.07535
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 딱 해지는 를을 저고 있다.           | 0.0000	False
과 축제 주점 등지에서 같은 과 여학생들을 언급하며 성희을 했고, | 1 롯해지 발 를을 한 있다.          | 0.0000	False
국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 딱 보호서열린다 일오 다.            | 0.0000	False
로스 대표(44 구속)와 공모해 주가조작으로 수십억대의 시세차익 | 와 여일 을 당이다.               | 0.0000	False
시한 결과다. 경북도와 구미시는 물론 가물막이 아래 준설공사를 담 | 날씨정보습 갖인고 하 기다자           | 0.0000	False
--------------------------------------------------------------------------------
[8900/100000] Train loss: 0.00007, Valid loss: 0.00454, Elapsed_time: 3254.95255
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
신선함을 생명으로 하는 흰 우유의 새로운 기준을 제시했다. 젖소도 | 남씨정으습갖  다자                | 0.0000	False
시 김종창 금감원장에게 청탁한 정황을 포착한 것으로 30일 확인 | 공 보해지안 조촉을 길 있다.          | 0.0000	False
인 불법사찰 국정조사, 대통령 내곡동 사저부지 특검 등 19대 국회 개 | 기상정보 전 것으로보인다."           | 0.0007	False
각종 모의시험 설비를 갖춘 원자력 서비스센터를 준공했다. 원자력 | 성비 게 10 원곤니다."            | 0.0000	False
김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 남북보해지 조촉을습 있다.            | 0.0000	False
--------------------------------------------------------------------------------
[9000/100000] Train loss: 0.00006, Valid loss: 0.00424, Elapsed_time: 3290.44972
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
각종 모의시험 설비를 갖춘 원자력 서비스센터를 준공했다. 원자력 | 빛바비 게를  드습니다."            | 0.0000	False
니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 성마비 자를 이경이 기자             | 0.0000	False
크 서비스를 규제하는 방안을 검토하다고 밝다. 데일리직 길인 | 1 해광 을 을세다 저이고 있다.        | 0.0000	False
베 시에서 20명으로 구성된 강도단이 30분만에 은행 3곳을 터는 사 | 성마 환위 해1 0촉각 곤두했다.        | 0.0000	False
흡수된다. 리스크검사지원국은 폐지된다. 금감원은 또 소비자서비스 | 날씨정보앞 니다.                 | 0.0010	False
--------------------------------------------------------------------------------
[9100/100000] Train loss: 0.00007, Valid loss: 0.00435, Elapsed_time: 3325.40776
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
수 있는 구조로 설계되어 있다. 이렇게 되면 견인성과 제동성이 클 뿐 | 단과 총치입면 조찰저 보이다.          | 0.0000	False
중소기업에 대해 근로시간 단축 지원금을 주고 있는데도 울산지역 | 을 켜일법방에 촉을 기자             | 0.0000	False
롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 남북정보산조을 고 있다.             | 0.0000	False
장경기록문화 테마파크 조성, 통영 국제음악당 건립, 김해 중소기업 | 10 을 열이 으갖고보호 기자          | 0.0000	False
다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 적문까지 확 확세된 것으종호 기자        | 0.0000	False
--------------------------------------------------------------------------------
[9200/100000] Train loss: 0.00006, Valid loss: 0.00443, Elapsed_time: 3361.06406
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한통운은 최근 어학능력과 국제 감각을 갖춘 '글로벌 인재 풀'을 강화 | 공 롯는지 를 니다."              | 0.0000	False
학과정을 운영하고 있으며, 지난해 약 2000여 명이 신청했을 정도로 | 귀다.                       | 0.0263	False
한은 금통화위원회에 차관을 참석시키는 열석발언권을 행사해 중 | 10롯해지에 앞짐 저 보있다.          | 0.0000	False
니다. 토요일과 일요일에도 날씨로 인한 큰 불편은 없을 것으로 보 | 빛 영일광 y린찰 사원보 니다.         | 0.0000	False
예상니다. 낮 기온은 서울 3도, 광주 8도, 부산은 10도까지 오르 | 날씨 해 다.자                  | 0.0004	False
--------------------------------------------------------------------------------
[9300/100000] Train loss: 0.00006, Valid loss: 0.00461, Elapsed_time: 3397.00672
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 성마 총보을 경찰린 사망하직도 기자       | 0.0000	False
근 노동현안에 대해 변화하는 만큼 믿고 지켜볼 것"이라고 말했다. 한 | 을 지며안처 을 저 고 있다.          | 0.0000	False
비정규직법안에 대한 국회 처리를 둘러싸고 여당과 노동계가 한 | 공 딱데텔에 촉을 기자              | 0.0000	False
니다. 날이 풀리면서 내일 오전에 미세먼지 농도가 높아질 수 있어 주 | 공 롯보호에 시린리 니다.            | 0.0000	False
여수에서는 현재 풍속이 초속 12.5미터로 관측되고 있고 대구 역시 | 빛동 동높광입니 를 기자             | 0.0000	False
--------------------------------------------------------------------------------
[9400/100000] Train loss: 0.00006, Valid loss: 0.00448, Elapsed_time: 3431.99237
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
급할 가능성이 높다. 3억원을 개인적으로 사용하지 않다면 모든  | 빛문 지영확 를 산짐 저이다.          | 0.0000	False
인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 10조을합1면 0조촉원을 고 있다.       | 0.0000	False
경영실적보고서를 평가편람 내용과 다르게 작성하지 말라"고 주의를 | 남과조광 찰다.yt이 사망기 기자        | 0.0000	False
여야 원내대표간 합의했던 등록금 부담완화 문제와 저축은행 국정조 | 을 지켜보며일  기자               | 0.0043	False
이 가동돼 한은의 통화정책이 영향을 받는 것 아니냐는 우려의 목소 | 공동 롯광 텔에 앞 기자             | 0.0000	False
--------------------------------------------------------------------------------
[9500/100000] Train loss: 0.00006, Valid loss: 0.00408, Elapsed_time: 3467.60365
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
고 더 큰 사회 문제를 일으키는 원인이  것"이라며 "결국 그들이 당 | 빛 환총를등 경이 기자              | 0.0000	False
국으로 확대 개편하는 등 조직을 대폭 확대한다. 특히 감독과 검사업 | 딱 지 을 확저 오종호 기자           | 0.0001	False
습니다. 오늘 경북 상주는 낮 기온 25도까지 올고 서울도 22.1도 | 귀국피치보  다자                 | 0.0001	False
지만 아들의 답안지 작성에 관여한 일이 없다"며 결백을 주장한 것으 | 입을합면 경열린다 사직오보 다.         | 0.0000	False
는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 입을열지 확산조당길 보호 기자          | 0.0000	False
--------------------------------------------------------------------------------
[9600/100000] Train loss: 0.00006, Valid loss: 0.00438, Elapsed_time: 3503.33269
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 적 높보광일해 해정다.              | 0.0000	False
물의를 빚어 감찰 대상에 올던 c검사가 이날 사표를 제출했다고 밝 | 입을 열지환 를당앞 기다자            | 0.0000	False
럴 플랜'를 출시했다. 옥수수를 주원료로 한 배합사료 대신 풀을 먹 | 남 롯등면 확인 이고 있다.           | 0.0000	False
상이한 광해방지 제도 및 적용기술에 대해 한국녹색산업진흥협회와 | 남 켜보등 y0 저 기다.            | 0.0000	False
는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 입을지 확산조당길 보 기.            | 0.0000	False
--------------------------------------------------------------------------------
[9700/100000] Train loss: 0.00006, Valid loss: 0.00469, Elapsed_time: 3538.32906
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
대피해 있다.                   | 대피해 있다.                   | 0.5978	True
1년 만에 재개습니다. cj그룹 이미경 부회장 퇴진을 압박한 니 | 을갈보습입니 환기0를 앞각길 곤달있다.     | 0.0000	False
전북도를 찾는 관광객에게 오랫동안 기억에 남을 추억거리를 선물할 | 남북갈을지면 세린 앞원곤 기자          | 0.0000	False
마당 등을 선보일 계획이다. 남원에선 5월 6일부터 10일까지 문화체 | 남북갈등을 최면다 경찰린 앞곤 기자       | 0.0000	False
성장위를 다루었으며, 명확한 임무가 부여되어 있다"고 말했다. 그는 | 정보 기자                     | 0.0417	False
--------------------------------------------------------------------------------
[9800/100000] Train loss: 0.00005, Valid loss: 0.00448, Elapsed_time: 3573.96187
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
지급률 수정 등 적절한 조치를 취하라"고 통보하고, 수공 사장에게 " | 의로지해 당저 만도 있다.            | 0.0000	False
다. 이 관계자는 이어 "전입학 당시 오 교사가 전입학서류를 담당 교 | 한 롯보위안 앞y 원곤달있다.          | 0.0000	False
성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 을 지켜보일 산1y 앞을 곤두니다.       | 0.0000	False
성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 기북갈보 산등을앞한 고 있다.          | 0.0000	False
들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 을 롯데호법 방에 앞리도두 기자         | 0.0000	False
--------------------------------------------------------------------------------
[9900/100000] Train loss: 0.00006, Valid loss: 0.00498, Elapsed_time: 3609.79972
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 남북정보산조을 고 있다.             | 0.0000	False
불 축제'를 연다.                | 불 축제'를 연다.                | 0.4283	True
건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 을 지보호안 열 린 앞각고 고 있다.      | 0.0000	False
밝다. 박씨도 로비 자체를 부인하고 있다. 박씨는 부산저축은행에 | 날보 기자                     | 0.0091	False
는 이 자리에서 "민주노총은 운명적으로 정권교체를 함께해야 할 파 | 을 지환안 를을니린 사앞원보니다"        | 0.0000	False
--------------------------------------------------------------------------------
[10000/100000] Train loss: 0.00005, Valid loss: 0.00422, Elapsed_time: 3644.72219
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
낮부터 기온이 오름세를 보이면서 추위가 금세 누그러지습니다. 그 | 성마 환를 해니1 0사원 다.          | 0.0000	False
안부 피해자들이 보다 적극적으로 자신의 명예와 존엄성을 지킬 수 | 와 동일여 를 기자                | 0.0001	False
단과 총격을 벌이다 경찰이 사망하기도 했다.  | 단과 총격을 벌이다 경찰이 사망하기도 했다.  | 0.3584	True
. 내일은 오늘보다 기온이 더 높습니다. 서울 아침 기온 영하 6도, | 남북을 합지 확산조된 사망하 고 있다.     | 0.0000	False
서울시 교육청 관계자 등은 20일 c군이 b고에 전입학한 작년 3월 가 | 남갈보 산을갖마 이원달이다.           | 0.0000	False
--------------------------------------------------------------------------------
[10100/100000] Train loss: 0.00006, Valid loss: 0.00610, Elapsed_time: 3680.27946
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
라 보험권역은 기존 2국 2실 체제에서 5개 국 체제로 대폭 확대 예 | 남정등으로 최세저 사것을 보 있다.       | 0.0000	False
박 전 대통령은 재산 몰수와 추징 시효가 10년으로 늘어난 전두환 특 | 을 지환광 광니y 0원 니다.          | 0.0000	False
성은 배제할 수 없을것으로 보여 향후 검찰의 수사가 주목된다. | 빛바랜 영광입니 당길고 있다.          | 0.0000	False
표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 빛 랜 산광산당 길 계획있다.          | 0.0000	False
경영실적보고서를 평가편람 내용과 다르게 작성하지 말라"고 주의를 | 남 랜조광 입 찰다.y사 사망기 기자      | 0.0000	False
--------------------------------------------------------------------------------
[10200/100000] Train loss: 0.00005, Valid loss: 0.00429, Elapsed_time: 3715.23470
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
검찰이 박근혜 전 대통령에 대한 재산 동결 조치를 취했습니다. 내곡 | 입 열 열시린 0촉을 두니다."         | 0.0000	False
법무부 공보관은 "김우중씨 명의로 된 한국 여권과 프랑스 여권을 모 | 공치보 지언 를 기자               | 0.0000	False
3건 25억7000여만원을 삭감했다. 여기에는 김 지사의 대표적인 공 | 와 환여일를 을당 앞사당 기자          | 0.0000	False
어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 |  지켜일안 을  기자               | 0.0000	False
창 올림픽은 88올림픽 이후 성장과 발전을 전세계에 알리는 좋은 계 | 소문이 광입1 앞한저 다자            | 0.0000	False
--------------------------------------------------------------------------------
[10300/100000] Train loss: 0.00005, Valid loss: 0.00426, Elapsed_time: 3750.79423
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
역으로 지목하고 있는 '다이아드 아일랜드'에 포함돼 있어 이 같은 | 의해지 2 박언를 니다."            | 0.0000	False
초 저질던 범죄보다 큰 사회적 비용이 들게 된다"고 말했다. 시민단 | 기북정보까 산 조저 보있다."          | 0.0000	False
로스 대표(44 구속)와 공모해 주가조작으로 수십억대의 시세차익 | 와 높일 갖속입이다.               | 0.0000	False
한국수자원공사의 물관리에 구멍이 뚫다. 두달새 같은 지점에서 유 | 공동 롯광위1 앞만길 고 있다.         | 0.0000	False
의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 정보니다.                     | 0.0705	False
--------------------------------------------------------------------------------
[10400/100000] Train loss: 0.00006, Valid loss: 0.00467, Elapsed_time: 3786.39121
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
아랍어, 베트남어 등 떠오르는 신흥 국가 언어강좌도 신청이 점차 늘 | 성문이 영격등을 면 원고 있다.         | 0.0000	False
정책 우선순위를 두는 가운데 고용회복이 지속 수 있는 방향으로 | 단과 총격을 세  기자              | 0.0000	False
와의 동일여부 등을 확인할 예정이다.      | 와의 동일여부 등을 확인할 예정이다.      | 0.0959	True
m의 눈이 내리습니다. 내일 비가 오는 지역에서는 돌풍과 함께 천 | 빛 켜보광입니 y린 앞 구속있다.        | 0.0000	False
톱스타 원이 오는 27일 일본 요코하마에서 자선 팬미팅을 연다. 원 | 날해 기자                     | 0.0194	False
--------------------------------------------------------------------------------
[10500/100000] Train loss: 0.00005, Valid loss: 0.00479, Elapsed_time: 3821.34684
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 남과총보을보합 찰이 고 다.           | 0.0000	False
계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 소문이 광입1 확데일대 세다.          | 0.0000	False
념이 충분히 반영돼 있다고 본다"고 주장했다. 또 "외통위를 통과해서 | 공 롯데호에 열린 장병종호 기자         | 0.0000	False
단과 총격을 벌이다 경찰이 사망하기도 했다.  | 단과 총격을 벌이면 경찰다 사망하기도 했다.  | 0.0045	False
대구지법 제 12형사부(재판장 김종필 부장판사)는 1일 생활고로 남 | 을 환일부  기자                 | 0.0000	False
--------------------------------------------------------------------------------
[10600/100000] Train loss: 0.00005, Valid loss: 0.00465, Elapsed_time: 3857.00481
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
습니다. 문 대통령은 또 평창 올림픽을 평화 올림픽으로 만들어, 북핵 | 와 동보일호 를을 한 보있다.          | 0.0000	False
신한카드는 5일 경기도 성남시 차의과학대학교에서 이재우 사장과 | 정보습 산원 기자                 | 0.0000	False
습니다. 서해와 남해안에서는 짙은 안개를 주의하야습니다. 비 | 빛의 환일에 을 니다.              | 0.0001	False
영장을 청구했으며 긴급체포돼 수감 중인 은 전 위원을 이날 다시 불 | 입을습까 소전앞이다.               | 0.0000	False
제 상황처럼 해볼 수 있게 돼정비 전문인력 양성은 물론, 새로운 정비 | 기북정보습 박촉전을보습니다."          | 0.0000	False
--------------------------------------------------------------------------------
[10700/100000] Train loss: 0.00006, Valid loss: 0.00475, Elapsed_time: 3892.55887
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
로스 대표(44 구속)와 공모해 주가조작으로 수십억대의 시세차익 | 와 일 을속이다.                 | 0.0000	False
한진중공업 김주익 열사와 두산중공업 배달호 열사 등 수많은 동지들 | 공 북리안부 을  기자              | 0.0000	False
행 총재는 15일 오전 서울 은행회관에서 조찬 간담회를 갖고 거시경 | 공 롯등호 전산세다 장원하호 기자        | 0.0000	False
자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 딱 해는 을 갖확저고 있다.           | 0.0001	False
고 했다.                     | 수 했다.                     | 0.3709	False
--------------------------------------------------------------------------------
[10800/100000] Train loss: 0.00005, Valid loss: 0.00445, Elapsed_time: 3927.49700
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
원은 옛 삼성동 자택을 67억 원에 팔면서 생긴 돈으로 파악하고 있습 | 북갈 기다.                    | 0.0001	False
예상니다. 낮 기온은 서울 3도, 광주 8도, 부산은 10도까지 오르 | 전로이 해니다."                 | 0.0087	False
수 기자                      | 수 기자                      | 0.9141	True
한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 씨정보 있다.                   | 0.0056	False
. 내일은 오늘보다 기온이 더 높습니다. 서울 아침 기온 영하 6도, | 남북을 열지 확산조된 사망하 고 있다.     | 0.0000	False
--------------------------------------------------------------------------------
[10900/100000] Train loss: 0.00004, Valid loss: 0.00375, Elapsed_time: 3963.08772
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 공 총격입 y린 앞직 곤 있다.         | 0.0000	False
다. 서울우유는 지난 11일 75주년 기념식에서 '행복한 젖소, 행복한 | 남0갈을까지 산 산짐마저 보이다 다.      | 0.0000	False
위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 빛바 일 일 전드습니다."            | 0.0000	False
나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 빛과 광위입 경찰이 기자             | 0.0000	False
로 평년기온을 11도 가량 웃돌면서 5월 상순에 해당되는 고온현상이 | 남북갈등으 확최세호 장병호 했다.        | 0.0000	False
--------------------------------------------------------------------------------
[11000/100000] Train loss: 0.00005, Valid loss: 0.00470, Elapsed_time: 3998.68211
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
업부서로 배치해 검사기능과 소비자보호 업무를 대폭 강화한다는 방 | 을 지데텔에 열서린. 촉각을 두세다.      | 0.0000	False
시 김종창 금감원장에게 청탁한 정황을 포착한 것으로 30일 확인 | 공 보해호안 방촉을 있다.            | 0.0000	False
위반)로 김모(18)군을 불구속 입건했다. 경찰에 따르면 김군은 고교 | 와 동여 을 확세 이정보습니다."        | 0.0000	False
서울시 교육청 관계자 등은 20일 c군이 b고에 전입학한 작년 3월 가 | 입을보조입 조원저고 이다.            | 0.0000	False
에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 적 높여광  다.                 | 0.0001	False
--------------------------------------------------------------------------------
[11100/100000] Train loss: 0.00005, Valid loss: 0.00445, Elapsed_time: 4033.49639
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
게 불며 쌀쌀하습니다. 주말에는 대체로 맑은 날씨가 예상니다. | 남북갈보1 산성을 있다.             | 0.0000	False
정부가 주40시간 근로제를 앞당겨 시행하면서 신규 인력을 채용하는 | 한 경찰이사고 했다.               | 0.0000	False
연합뉴스와의 전화 통화를 통해 "최근 변호사인 친동생(김형진 씨)이 | 입을리 해다.                   | 0.0001	False
생들이 전입학을 요청하면 시 교육청에서 학교를 배정하고 해당 학교 | 단과 영보광를입  기자              | 0.0000	False
으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 빛 켜보광 입  기자               | 0.0000	False
--------------------------------------------------------------------------------
[11200/100000] Train loss: 0.00005, Valid loss: 0.00442, Elapsed_time: 4069.39038
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
이 과제가 국가 추진과제로 선정다. 이에 따라 오는 2014년을 목 | 남 켜환등 당면 저이 다.            | 0.0000	False
기 정보를 확인하야습니다. 오늘 낮 최고 기온은 서울 2도 등 어 | 1 롯해텔1에 0을 바 기자           | 0.0000	False
쟁력 강화 등 내수기반 강화가 필요하다"는 데 의견을 같이했다. 양 | 공동 롯데호면 0찰에 직각오 곤 있다.     | 0.0000	False
생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 전해해습니 있다.                 | 0.0001	False
로 했던 민주노총은 오후 11시 투쟁본부회의를 열어 총파업 방침에 | 단다는치 한 바 있다.              | 0.0000	False
--------------------------------------------------------------------------------
[11300/100000] Train loss: 0.00005, Valid loss: 0.00447, Elapsed_time: 4104.36303
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
상태로 이뤄지는 콜드체인시스템을 도입해 2000여 곳의 전용목장에 | 성과랜광위1 앞 계 기자             | 0.0000	False
반성이 있어야 한다"고 말했다. 이에 대해 문 후보측 김경수 공보특보 | 입을보입 처y서 앞곤 기자            | 0.0000	False
들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 입 지보열호 열세린 저 보하기도 있다.     | 0.0000	False
면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 남 해광 산짐다 t 고 있다.          | 0.0000	False
로 정하고 추진했다. 이들 사업에 도는 사업당 200억 원을 한도로 20 | 1 보만방에  기자                | 0.0000	False
--------------------------------------------------------------------------------
[11400/100000] Train loss: 0.00005, Valid loss: 0.00611, Elapsed_time: 4140.03412
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
남북갈등으로까지 확산 조짐마저 보이고 있다.  | 남북갈등으로까지 확산 조짐마저 보이고 있다.  | 0.5374	True
태가 많이 쓰이기도 한다. 반면 스노우 타이어 트레드는 깊이가 깊고 | 빛 보 y y찰이 보 다.            | 0.0000	False
유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 단 총보 경이 이 기자              | 0.0000	False
국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 딱 보호서린다 장각오 다.            | 0.0000	False
기 오렌지 등 풍부한 과실향과 튼튼한 구조감을 가진 리포니아 최 | 을 보해법 를을 계획이다.            | 0.0000	False
--------------------------------------------------------------------------------
[11500/100000] Train loss: 0.00004, Valid loss: 0.00427, Elapsed_time: 4175.62716
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
는 "노동계가 기대가 던 데 반해 참여정부가 한계가 있었다는 점을 | 남을보까면 환시리다.               | 0.0000	False
이는 방식을 택해 공장식 축산을 할 수 없다며 한정생산을 강점으로 | 1 해지 를을 저 앞정전 있다.         | 0.0000	False
"내일이면 평창 동계 올림픽이 개막합니다. 내일 평창은 대체로 구름 | 빛바랜 영광입니 열다 사원종호 기자       | 0.0000	False
유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 성마 총등을 벌경찰 사망하도 기자        | 0.0000	False
정이다. 은행권역에서는 은행감독국이 신설되고 외환업무실이 외환 | 성 켜일며법에 을 면 사원곤세기다자       | 0.0000	False
--------------------------------------------------------------------------------
[11600/100000] Train loss: 0.00005, Valid loss: 0.00440, Elapsed_time: 4210.47154
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
불러다. 수자원공사는 지난 5월 사고 당시 다시는 사고가 발생하지 | 딱다.                       | 0.0425	False
지만 아들의 답안지 작성에 관여한 일이 없다"며 결백을 주장한 것으 | 입보을합면 경열린 직오곤 세다.         | 0.0000	False
어우러져 있는 그랑 마레농은 장어의 느끼한 맛을 잘 잡아준다. 또한 | 빛 갈보광 전짐다 사이고 기다자         | 0.0000	False
성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 기북보까지 산 를을 바 있다.          | 0.0000	False
과 함께 마시면 더욱 좋다. 열대과일의 풍미와 산도, 미네이 잘 살아 | 성마비환 이다.                  | 0.0006	False
--------------------------------------------------------------------------------
[11700/100000] Train loss: 0.00005, Valid loss: 0.00437, Elapsed_time: 4246.01598
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 단다 총격최 확산를을갖당한길 계획있다.     | 0.0000	False
다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 소까지 확 최세된 것으보호 다.         | 0.0000	False
통해 금감원 검사에서 '강도와 제재수준을 완화해달라'는 취지로 당 | 공 조을열지다 0방앞 곤호 기자         | 0.0000	False
회복을 중심으로 거시정책이 운영돼야 한다는 공통인식에 따라 정책 | 남정등보까지확산조 앞이있다.           | 0.0000	False
다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 기북갈보 합10 사하도 있다.          | 0.0000	False
--------------------------------------------------------------------------------
[11800/100000] Train loss: 0.00004, Valid loss: 0.00451, Elapsed_time: 4281.61789
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
니다. 현재 중부 지방과 영남을 중심으로 건조 특보가 확대되고 있습 | 와 동일여에 을 바호 기자            | 0.0000	False
주소로돼 있어야 하고 실제 거주해야 한다고 교육청 관계자가 전했 | 딱 해지면 환를 를열된 고보세다.        | 0.0000	False
의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 와 동일 촉  기자                | 0.0000	False
치행위를 당장 그만두고 이명박 대통령의 불법대선자금에 연루된 이 | 북정보습확 박앞한 있다.             | 0.0000	False
총장 직속으로 외부 전문가가 참여하는 가칭 '성희성폭력성차별 자 | 남북등보 까최산세린 앞원전달있다.        | 0.0000	False
--------------------------------------------------------------------------------
[11900/100000] Train loss: 0.00004, Valid loss: 0.00411, Elapsed_time: 4316.50082
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
크 서비스를 규제하는 방안을 검토하다고 밝다. 데일리직 길인 | 1 해광 을 을세다 저 고보 있다.       | 0.0000	False
입니다. 다만 제주도는 일요일 오후부터 다시 비가 오기 시작하습 | 빛바" 해 입저 망고 있다.           | 0.0000	False
역으로 지목하고 있는 '다이아드 아일랜드'에 포함돼 있어 이 같은 | 의해보지 2 언를 니다."            | 0.0000	False
wonbin'을 열어 팬들과 만난다. 약 5천여명의 팬들이 참석할 예정 | 남북갈등보습합면 환시0 조원이 이다.      | 0.0000	False
다. 대구 최세호 장병호 기자          | 다. 대구 최세호 장병호 기자          | 0.0843	True
--------------------------------------------------------------------------------
[12000/100000] Train loss: 0.00004, Valid loss: 0.00411, Elapsed_time: 4352.04977
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
입장을 발표하고, 사회적으로 손가락질받을 일이 학교에서 발생해 안 | 기북갈 산습 다.                 | 0.0001	False
소재로 한 '2011 고창 복분자 푸드페스티벌'을 개최해 축제의 계절에 | 한 지 광 한 기자                | 0.0000	False
법을 "최악의 사태를 몰아오는 정치적 도발"로 규정했다. 북한은 논평 | 성마랜 광등을y 를 기다자            | 0.0000	False
고 있다. 야당과 시민사회단체들은 중형을 선고한 것은 지나치다고 | 한의 지 를을니다.                | 0.0001	False
는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 심등을입니면 0원곤두니다.            | 0.0000	False
--------------------------------------------------------------------------------
[12100/100000] Train loss: 0.00004, Valid loss: 0.00424, Elapsed_time: 4387.69348
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 딱 해지는 을 갖확저고 있다.          | 0.0001	False
인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 10조을해1면 0촉원을 고 있다.        | 0.0000	False
9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 남북갈보까 확산세 박촉각보호니다."       | 0.0000	False
면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 남북갈보로 지세 장보 있다.           | 0.0000	False
에 오른다. 이 기간 동안 방단은 블라디보스 한인 집단 거주지, 안 | 소문이지 확산를 있다."             | 0.0001	False
--------------------------------------------------------------------------------
[12200/100000] Train loss: 0.00004, Valid loss: 0.00595, Elapsed_time: 4422.60894
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 날 정보습 전니다.                | 0.0006	False
한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 공 켜보호 언 바 다.              | 0.0000	False
하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 기북보안 산등을 있다.              | 0.0000	False
입니다.                      | 이다.                       | 0.8284	False
다. 방 기간 동안 중국 및 러시아 현지 대학 교수들의 특강과 토론도 | 빛마비 를광등을 확경 기자            | 0.0000	False
--------------------------------------------------------------------------------
[12300/100000] Train loss: 0.00004, Valid loss: 0.00463, Elapsed_time: 4458.11637
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
비 부담을 덜어주기 위해 주40시간제를앞당겨 시행하면서 정규직을 | 한 보격 지1치 를  기자            | 0.0000	False
일부 보도에 대해 "사실이 아니다"라고 밝다. 김형곤은 14일 오후 | 입을 지최호 린. 고니다.            | 0.0000	False
파도가 높고 날이 어두워지자 이날 구조작업은 철수하고 17일 오전 | 공 켜보호법에 처방에 촉각직 기자        | 0.0000	False
건축학과를 졸업하고 현재중공업분야에서 해외 수출 업무를 전담하 | 1 갈등으 산 했다자               | 0.0000	False
회복을 중심으로 거시정책이 운영돼야 한다는 공통인식에 따라 정책 | 남북등으보로까지확산조 앞이있다.         | 0.0000	False
--------------------------------------------------------------------------------
[12400/100000] Train loss: 0.00004, Valid loss: 0.00416, Elapsed_time: 4492.98482
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
라 보험권역은 기존 2국 2실 체제에서 5개 국 체제로 대폭 확대 예 | 남북등을로 지1호 촉원을 있다.         | 0.0000	False
로 분리할 예정이다. 금감원은 조직개편과 함께 지원부서 인력을 현 | 딱 지켜보일에 앞을  다.            | 0.0000	False
경영실적보고서를 평가편람 내용과 다르게 작성하지 말라"고 주의를 | 남정등갖 산원하다 고 기자            | 0.0000	False
담임교사로부터 기말시험 답안지 대리작성 도움을 받은 현직 검사의 | 을 보호1린 0방 촉각을 곤달세다.       | 0.0000	False
결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 성마영일해 병호 기자               | 0.0000	False
--------------------------------------------------------------------------------
[12500/100000] Train loss: 0.00004, Valid loss: 0.00486, Elapsed_time: 4528.65239
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
니다. 검찰은 유죄 확정 시 범죄수익을 차질 없이 추징하기 위해 재산 | 와 총보광 y 이 다자              | 0.0000	False
다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 소문까지 확 최세된 것으종호 다자        | 0.0000	False
두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 소문이 확한 이다.                | 0.0001	False
검찰이 박근혜 전 대통령에 대한 재산 동결 조치를 취했습니다. 내곡 | 입을 열 소열시린 사리을 했다.         | 0.0000	False
다. 차장검사 출신으로 부산저축은행의 고문 변호사던 박 모씨는 | 와 켜보등 산찰y 사고 있다.          | 0.0000	False
--------------------------------------------------------------------------------
[12600/100000] Train loss: 0.00004, Valid loss: 0.00450, Elapsed_time: 4564.20276
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
보했지만 c검사는 "말미를 달라"며 출석을 미뤄온 것으로 알려다. | 빛바 랜동 일부 을한 바 있다.         | 0.0001	False
건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 남북정보까환 조당 길 계있다.          | 0.0000	False
건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 남 북보환 를앞 있다.              | 0.0000	False
성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 기보 산등을앞한 고 있다.            | 0.0000	False
고 있는 추세라는 것이 회사 측 설명이다. 사이버 강의는 스마트으 | 을갈으 지 산입저  기자             | 0.0000	False
--------------------------------------------------------------------------------
[12700/100000] Train loss: 0.00004, Valid loss: 0.00452, Elapsed_time: 4599.17388
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
법사위에 계류돼 있는 법안에 북한에 대한 인도적인 지원 내용이 들 | 소 이데광산1 저리 다.             | 0.0000	False
비정규직법안에 대한 국회 처리를 둘러싸고 여당과 노동계가 한 | 공 딱해텔에 촉을 기자              | 0.0000	False
건축학과를 졸업하고 현재중공업분야에서 해외 수출 업무를 전담하 | 1북 갈등으 고산 기다자             | 0.0000	False
국정원 뇌물 사건을 맡은 유영하 변호사가 보관하고 있습니다. 유 변 | 날씨총보을   기자                | 0.0000	False
년 가까이 끌어온 북한인권법을 이번 임시국회에서 반드시 처리하 | 성과 총보광 산찰 경찰이 기다자         | 0.0000	False
--------------------------------------------------------------------------------
[12800/100000] Train loss: 0.00004, Valid loss: 0.00422, Elapsed_time: 4634.85982
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한인권법을 놓고 하게 맞서는 동안 북한은 이같은 움직임에 대한 | 북갈보산앞을한 있다.               | 0.0001	False
수 있는 구조로 설계되어 있다. 이렇게 되면 견인성과 제동성이 클 뿐 | 날북갈으보습입다 경찰이경이 원니다.       | 0.0000	False
니다. 앞으로 대기는 점점 더 건조해 질 것으로 보여 화재가 나지 않도 | 정보다."                     | 0.1425	False
롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 남북정보습산조을 고 있다.            | 0.0000	False
후배가 되는 것을 싫어하는 것을 보고 아내의 지인이 사는 강동구로 | 한 기자                      | 0.3213	False
--------------------------------------------------------------------------------
[12900/100000] Train loss: 0.00004, Valid loss: 0.00471, Elapsed_time: 4670.41553
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
에 오른다. 이 기간 동안 방단은 블라디보스 한인 집단 거주지, 안 | 소문이지 확산를 있다."             | 0.0000	False
9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 남북정보까 확산 앞 호 기자           | 0.0000	False
"문재인 대통령이 남북 문제와 관련해 강력한 국방력을 기반으로 대 | 빛 지일해 산세 것장으로보다.          | 0.0000	False
의했기 때문에 필요할 경우 c검사는 참고인 자격으로 소환 가능 | 을 동며광입에 y찰이 사고 있기자        | 0.0000	False
법을 "최악의 사태를 몰아오는 정치적 도발"로 규정했다. 북한은 논평 | 빛바랜 영광입니 길 바있다.           | 0.0000	False
--------------------------------------------------------------------------------
[13000/100000] Train loss: 0.00004, Valid loss: 0.00445, Elapsed_time: 4705.35073
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 공 롯을합서 열린다 장병호 기자         | 0.0000	False
사용할 계획이다. 이재우 사장은 "이번 협약 체결로 병마와 싸우고 | 입갈등입열 산열하 저 이다.           | 0.0000	False
다. 문 대통령은 오늘 대한노인회 회장단과 가진 오찬에서 과거처럼 | 남북갈등을 벌인 경찰이 기자           | 0.0000	False
동 자택과 1억 원짜리 수표 30장 등 모두 58억 원 규몹니다. cj그룹 | 공 보광입니 으을  기자             | 0.0000	False
념이 충분히 반영돼 있다고 본다"고 주장했다. 또 "외통위를 통과해서 | 딱 보입 앞 구속했다.              | 0.0000	False
--------------------------------------------------------------------------------
[13100/100000] Train loss: 0.00004, Valid loss: 0.00427, Elapsed_time: 4741.05456
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 10조을 열면 조앞한저 바이 기자        | 0.0000	False
영장을 청구했으며 긴급체포돼 수감 중인 은 전 위원을 이날 다시 불 | 입을열면 시조 앞 계있다.            | 0.0000	False
학과정을 운영하고 있으며, 지난해 약 2000여 명이 신청했을 정도로 | 귀 있다.                     | 0.0297	False
황토현 동학축제'를 통해 1박2일 황토현숙영캠프, 동학농민명군 | 와 동광위1 언 기자               | 0.0000	False
초 저질던 범죄보다 큰 사회적 비용이 들게 된다"고 말했다. 시민단 | 남북정보까지산를 앞 두 다.자          | 0.0000	False
--------------------------------------------------------------------------------
[13200/100000] Train loss: 0.00004, Valid loss: 0.00423, Elapsed_time: 4776.75715
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 빛 총보등보 산를 당 있다.           | 0.0000	False
비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 의 지일 을앞당 계종호 기자           | 0.0000	False
의 집으로 옮기도록 주선했다는 일부 언론의 보도를 전면 부인한 것 | 10조을 지난2일박를을 바속했다.        | 0.0000	False
표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 공 지켜동 일발 를을 한 있다.         | 0.0000	False
술함을 여실히 드러다는 비판의 목소리가 높다. 5월 해평취수장의 | 빛마 랜환영를 광입니 린 사직을 보 했다.   | 0.0000	False
--------------------------------------------------------------------------------
[13300/100000] Train loss: 0.00004, Valid loss: 0.00423, Elapsed_time: 4811.75299
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
이 곧 글로벌 기업이  수 있다는 의지가 담겨있다고 회사측은 설명 | 남갈등까 짐마저 보이보있다."          | 0.0000	False
의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 날 해보습 기자                  | 0.0004	False
태가 많이 쓰이기도 한다. 반면 스노우 타이어 트레드는 깊이가 깊고 | 빛 보 y y이 보 기다.            | 0.0000	False
한국노총도 법안 강행시 벌이기로 했던 노사정위 탈퇴와 대정부 투쟁 | 남북갈으로지 조짐. 사만직 호기자        | 0.0000	False
을 위해폭동 가담자들을 엄벌해야 한다는 의견도 하다. 소 네 | 적 비환 를 을 1이 앞 바했다.        | 0.0000	False
--------------------------------------------------------------------------------
[13400/100000] Train loss: 0.00004, Valid loss: 0.00426, Elapsed_time: 4847.37496
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
법사위에 계류돼 있는 법안에 북한에 대한 인도적인 지원 내용이 들 | 한 영데광산1 앞저다.              | 0.0000	False
귀국치 않다.                   | 귀국치 않다.                   | 0.5075	True
럭으로 길목을 차단해 경찰의 접근을 막으며 3대의 픽업과 1대의 | 10을지 조을 한 바 있다.           | 0.0000	False
두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 적정보 갖인 원니다."              | 0.0000	False
감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 단과 총격해이정보습니다."            | 0.0000	False
--------------------------------------------------------------------------------
[13500/100000] Train loss: 0.00004, Valid loss: 0.00458, Elapsed_time: 4882.19465
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
나으며 한국노총과 민주노총 간부들도 회의장 안팎에서 진행상황 | 날 보광 등을 고있다.              | 0.0000	False
후배가 되는 것을 싫어하는 것을 보고 아내의 지인이 사는 강동구로 | 한 기자                      | 0.3231	False
상이한 광해방지 제도 및 적용기술에 대해 한국녹색산업진흥협회와 | 남 갈보등 y0  기다.             | 0.0000	False
황토현 동학축제'를 통해 1박2일 황토현숙영캠프, 동학농민명군 | 적 총격을해지 0만앞길 곤 했다.        | 0.0000	False
의했기 때문에 필요할 경우 c검사는 참고인 자격으로 소환 가능 | 을 롯며광입에 y찰이 사고 있기자        | 0.0000	False
--------------------------------------------------------------------------------
[13600/100000] Train loss: 0.00004, Valid loss: 0.00389, Elapsed_time: 4917.70969
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 입을열지 확산조당길 보호 기자          | 0.0000	False
을 보충하는데 도움을 준다. 게다가 와인이 식욕을 워주기 때문 | 적로씨는 해  0 정오습니다."         | 0.0000	False
충격패를 당하며 조 2위 자리가 위태로워지기도 했습니다.시리아와 | 남북정보까지환산 를앞린 앞고 있다.       | 0.0000	False
따라 비정규직법안 강행 처리시 24일 오전 8시부터 총파업을 벌이기 | 기 상정보 y0 사원곤 다.           | 0.0000	False
은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 을 지켜며법 1 앞을곤 기.자          | 0.0000	False
--------------------------------------------------------------------------------
[13700/100000] Train loss: 0.00004, Valid loss: 0.00495, Elapsed_time: 4953.26221
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
어에서 주로 볼 수 있는 패으로 타이어 좌우로 이 패여 있는 모양 | 단 총격등을 조치 산를 바니다."        | 0.0000	False
자심문(영장실질심사)은 29일 오전 서울중앙지법에서 열린다. 앞서 | 와의 동일광등을 한 고 기자           | 0.0000	False
행 총재는 15일 오전 서울 은행회관에서 조찬 간담회를 갖고 거시경 | 북씨정보 다.                   | 0.0007	False
블록패을 띄고 있다. 자동차가 눈길을 주행할 때 바퀴가 눈을 누르 | 남북갈등로 산전앞한고 기자            | 0.0000	False
오씨 학급만 공교롭게도 34명이어서 c군이 이 학급에 배정다고 해 | 대 상등보 지면마 0조원보고있다.        | 0.0000	False
--------------------------------------------------------------------------------
[13800/100000] Train loss: 0.00003, Valid loss: 0.00468, Elapsed_time: 4988.27584
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
정이다. 은행권역에서는 은행감독국이 신설되고 외환업무실이 외환 | 와 켜일며법에 을 면저 사원곤세기다자      | 0.0000	False
심야까지 회의를 열지 못하고 24일로 연기했다. 법안심사소위 위원 | 빛마 총광 입고 다.               | 0.0000	False
사용할 계획이다. 이재우 사장은 "이번 협약 체결로 병마와 싸우고 | 입을로이 전 이다.                | 0.0006	False
다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 입문까지 확세호 것병종호 기자          | 0.0000	False
결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 성마영일해드 계병호 기자             | 0.0000	False
--------------------------------------------------------------------------------
[13900/100000] Train loss: 0.00003, Valid loss: 0.00471, Elapsed_time: 5023.78428
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
에서 개봉한다. 일본 산이스포츠는 21일 "'마이 브라더('우리형'의 | 1총보 기자                    | 0.0011	False
단과 총격을 벌이다 경찰이 사망하기도 했다.  | 단 총격을 벌이다 경찰이 사망하기도 했다.   | 0.0945	False
것"이라며 "이 두 가지 임무에 가시적인 진전이 있기를 희망한다"고 | 단다 랜격호 호시기 고달세다.          | 0.0000	False
인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 을 보광입 y린 앞을 곤 있다.         | 0.0000	False
글을 올다는 의로 기소된조던 블쇼(20)와 페리 서트클리프 키 | 입정으습까 산조앞짐저 전보습 기자        | 0.0000	False
--------------------------------------------------------------------------------
[14000/100000] Train loss: 0.00004, Valid loss: 0.00441, Elapsed_time: 5059.39174
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 단 총보 경이 기자                | 0.0000	False
데 인식을 같이했다"고 밝다. 양 기관은 또 "거시정책은 물가안정에 | 입다 찰이 앞보 보고 있다.           | 0.0000	False
인하대는 의예과 남학생들의 여학생 집단 성희과 관련해 9일 공식 | 성해 환일를 을당서 사원 했다자         | 0.0000	False
후배가 되는 것을 싫어하는 것을 보고 아내의 지인이 사는 강동구로 | 한 기자                      | 0.2952	False
결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 성영일해드 병 기자                | 0.0000	False
--------------------------------------------------------------------------------
[14100/100000] Train loss: 0.00003, Valid loss: 0.00440, Elapsed_time: 5094.27042
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
찾는 이들이 많다. 하지만 여름 보양식에 와인을 곁들이면 더욱 좋다 | 딱 롯보며법입 앞 기자              | 0.0000	False
파클을 함께 마시면 입안에서 청량감이 살아나 더운 날씨에 처진 | 남북보y1y 앞원이 고 이다.          | 0.0000	False
시한 결과다. 경북도와 구미시는 물론 가물막이 아래 준설공사를 담 | 정보습산조을 확인저 예정이다.          | 0.0001	False
여야 원내대표간 합의했던 등록금 부담완화 문제와 저축은행 국정조 | 딱 보일 전해드 보호다"             | 0.0001	False
한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 1 보일에 를리 장곤 기자            | 0.0000	False
--------------------------------------------------------------------------------
[14200/100000] Train loss: 0.00004, Valid loss: 0.00454, Elapsed_time: 5129.87490
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
습니다. 오늘 경북 상주는 낮 기온 25도까지 올고 서울도 22.1도 | 한 총해격 있다.                 | 0.0000	False
사 사장은 연임이 유력한 것으로 알려다. 공기업 경영평가에서 a등 | 남 켜보입에 앞당길 계니다.           | 0.0000	False
말을 인용, 로르 회장이 한국에서사업을 위해 김 전 회장을 고문역으 | 남북갈등으로지면 조직 오종호 기자        | 0.0000	False
위반)로 김모(18)군을 불구속 입건했다. 경찰에 따르면 김군은 고교 | 공 롯1에 을열서다 사만직오호 기자       | 0.0000	False
조 파업 때로 거슬러 올라 간다. 문 후보는 당시 청와대 민정수석으로 | 날씨보 전 다.                  | 0.0001	False
--------------------------------------------------------------------------------
[14300/100000] Train loss: 0.00003, Valid loss: 0.00435, Elapsed_time: 5165.45628
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
개그맨 김형곤이 다이어트 테마 섬 개발을 위해무인도를 구입했다는 | 을 지보며안 처안 언앞을 있다.         | 0.0000	False
전혀 없었다"고 말했다. 앞서 프랑스 일간 리베라시은 로르 회장의 | 입을보 광 를만앞  기자             | 0.0000	False
른 자료를 사용했기 때문으로 밝혀다. 감사원은 기재부장관에게 " | 공 데호에 열린면 0만직 곤두했다.       | 0.0000	False
상이라 모욕했다. 이에 대해 일본군 위안부 피해자인 이용수 할머니 | 딱해지는 산만을 바 있다.            | 0.0000	False
자심문(영장실질심사)은 29일 오전 서울중앙지법에서 열린다. 앞서 | 와 동보광에 y. 를당 기자           | 0.0000	False
--------------------------------------------------------------------------------
[14400/100000] Train loss: 0.00003, Valid loss: 0.00487, Elapsed_time: 5200.37726
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
유동업조합으로 시작한 서울우유는 이후 국내 낙농산업의 신을 선 | 남북정보 산조갖 전 있다.            | 0.0000	False
조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 빛 총보등보 산를  있다.            | 0.0000	False
고 거듭 밝다. 김형곤의 동생이 산 암태면 벌목도는 신안군내 753 | 기북정보습 1이 사정보습니다."         | 0.0000	False
내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 한다 일 를로 보속다.              | 0.0000	False
단과 총격을 벌이다 경찰이 사망하기도 했다.  | 단과 총격을 벌이다 경찰이 사망하기도 했다.  | 0.4882	True
--------------------------------------------------------------------------------
[14500/100000] Train loss: 0.00003, Valid loss: 0.00501, Elapsed_time: 5235.87829
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 빛 총보  전원전속세다.             | 0.0000	False
전북도를 찾는 관광객에게 오랫동안 기억에 남을 추억거리를 선물할 | 남북갈을 지최 열린 앞원종 기자         | 0.0000	False
정비기술 향상, 전문인력 양성을 위해 면적 1375m2, 높이 25 미터 | 남정보 기자                    | 0.0479	False
국이 신설되고, 일부 실이 국으로 승격돼 기능이 강화된다. 13일 금 | 기정보습 니다."                 | 0.0099	False
서울우유가 치열한 유업체들간 경쟁시장에 '행복' 발을 들고 나 | 소문 확해로 지 사정보니다."          | 0.0000	False
--------------------------------------------------------------------------------
[14600/100000] Train loss: 0.00003, Valid loss: 0.00413, Elapsed_time: 5270.76806
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
하는 점에 비 b고교의 전입학 시스템에 중대 허점이 있었던 사실도 | 전보문 확까 짐마저 고 있다.          | 0.0000	False
발견, 경찰에 신고 했다. 이씨는 경찰에서 "집에 돌아와 보니 함께 지 | 남 갈등을 합찰이 사망원호 다.         | 0.0000	False
다. 하지만 민주노총은 아직 마음의 문을 열지 않다. 참여정부 노동 | 한다" 다.                    | 0.0007	False
운영할 필요가 있다는 데 공감했다"며 "특히 수출과 내수간 격차, 지 | 입 열호면 소열린 앞길 고 있다.        | 0.0000	False
자심문(영장실질심사)은 29일 오전 서울중앙지법에서 열린다. 앞서 | 와의 동일격등을 한 사 기다자          | 0.0000	False
--------------------------------------------------------------------------------
[14700/100000] Train loss: 0.00003, Valid loss: 0.00431, Elapsed_time: 5306.28242
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
술함을 여실히 드러다는 비판의 목소리가 높다. 5월 해평취수장의 | 빛마 환를 위입1 사직을 했다.         | 0.0000	False
주민의 인권을 실질적으로 증진하고 국제적 기준에 따라 인도적 지원 | 정 기자                      | 0.1672	False
의했기 때문에 필요할 경우 c검사는 참고인 자격으로 소환 가능 | 빛바랜 영광 저 원고 있다.           | 0.0000	False
급할 가능성이 높다. 3억원을 개인적으로 사용하지 않다면 모든  | 소문 지난광 를앞한 고 기자           | 0.0000	False
인하대는 의예과 남학생들의 여학생 집단 성희과 관련해 9일 공식 | 성해 환일를 을당 사원능 다자          | 0.0000	False
--------------------------------------------------------------------------------
[14800/100000] Train loss: 0.00003, Valid loss: 0.00434, Elapsed_time: 5341.69788
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
이들 중 15학번 남학생 9명은 주점에 후배 남학생들을 불러 동료 여 | 을 롯호서 린 촉오곤 기자            | 0.0000	False
된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 정해 다.                     | 0.0732	False
아랍어, 베트남어 등 떠오르는 신흥 국가 언어강좌도 신청이 점차 늘 | 성문이 북등을 면 경 원고 있다.        | 0.0000	False
그러지습니다. 바다의 물결은 동해상에서 최고 2.5미터까지 비교 | 1 켜일광 을  기자               | 0.0000	False
차량에 많이 쓰인다. 반면 래그형 은 트럭이나 버스에 장착되는 타이 | 남정보 다자                    | 0.0037	False
--------------------------------------------------------------------------------
[14900/100000] Train loss: 0.00003, Valid loss: 0.00431, Elapsed_time: 5376.53512
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
는 이 자리에서 "민주노총은 운명적으로 정권교체를 함께해야 할 파 | 을 지환안 를광을니린 사앞원습니다"       | 0.0000	False
위반)로 김모(18)군을 불구속 입건했다. 경찰에 따르면 김군은 고교 | 공 환텔에 을열서다 사만직오호 기자       | 0.0000	False
는데 회동 장소 중 한 곳이 서울이었다고 전했다. 1999년 10월 출 | 기정보 기자                    | 0.1027	False
념이 충분히 반영돼 있다고 본다"고 주장했다. 또 "외통위를 통과해서 | 남 보입 앞 구 있다.              | 0.0000	False
단 이씨와의 전화통화를 통해 이씨가 화재객차에 타고 있었고, 용의 | 성 총격 기자                   | 0.0006	False
--------------------------------------------------------------------------------
[15000/100000] Train loss: 0.00003, Valid loss: 0.00433, Elapsed_time: 5412.09335
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
검찰이 박근혜 전 대통령에 대한 재산 동결 조치를 취했습니다. 내곡 | 입을 열 소열린 사리을 있다.          | 0.0000	False
유동업조합으로 시작한 서울우유는 이후 국내 낙농산업의 신을 선 | 남북정보 전산조갖갖고 고전 있다.        | 0.0000	False
적 코리코 303호(860t급)의 선원 7명 가운데 이날 오후 7시 40분께 | 빛 지환를 을면 0조원형원입니다.        | 0.0000	False
해주지 않으면 주가조작 사실을 금감독원에 신고하다"고 협박한 | 딱보  기자                    | 0.0128	False
아랍어, 베트남어 등 떠오르는 신흥 국가 언어강좌도 신청이 점차 늘 | 단 동일광 등을 인저할고 있다.         | 0.0000	False
--------------------------------------------------------------------------------
[15100/100000] Train loss: 0.00003, Valid loss: 0.00478, Elapsed_time: 5447.68252
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
구조를 재개키로 했다. 한편 이날 좌초된 선박은 부산에서 석회석을 | 날정보다.                     | 0.0094	False
현영희 새누리당 의원의 공천헌금 의혹을 수사 중인 검찰이 현기환 | 공 동일법 를을 획있다.             | 0.0000	False
성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 빛바랜 광1다 사원 다.             | 0.0000	False
주 고등법원 순국장소인 순 감옥 등 중국, 러시아의 안중근 의사가 | 딱 켜보지 를을  있다.             | 0.0000	False
인 피고인이 잘못을 뉘우치며 정신적 후유증에 시달리고 있는 점 등 | 성마 총보격를 다자                | 0.0000	False
--------------------------------------------------------------------------------
[15200/100000] Train loss: 0.00003, Valid loss: 0.00445, Elapsed_time: 5482.51454
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
어지기를 바란다"고 말했다. 한편, 신한카드는 이날 차병원에 소아뇌 | 1 원입 y린기 사정로 보니다.         | 0.0000	False
니다. 앞으로 대기는 점점 더 건조해 질 것으로 보여 화재가 나지 않도 | 입해니 를를니다."                | 0.0000	False
행 총재는 15일 오전 서울 은행회관에서 조찬 간담회를 갖고 거시경 | 북씨정보 기자                   | 0.0007	False
에서 개봉한다. 일본 산이스포츠는 21일 "'마이 브라더('우리형'의 | 빛 총격 해 원고 있다.             | 0.0000	False
효중인 가운데 물결이 거세게 일습니다. 내일 아침에는 서울의 기 | 빛 총광등을1면 열조이. 사형원입니다.     | 0.0000	False
--------------------------------------------------------------------------------
[15300/100000] Train loss: 0.00003, Valid loss: 0.00500, Elapsed_time: 5518.12969
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
사진을 위주로, 최근 일본 개봉 스줄이 나온 원 주연의 '우리형' | 심 을갖 확인할 원이다."            | 0.0000	False
국이 신설되고, 일부 실이 국으로 승격돼 기능이 강화된다. 13일 금 | 기상정보습 니다."                | 0.0107	False
정보습니다."                   | 정보습니다."                   | 0.4926	True
법무부 공보관은 "김우중씨 명의로 된 한국 여권과 프랑스 여권을 모 | 공총보 지언 를 기자               | 0.0000	False
영국 폭동 와중에 페이스북에 선동 글을 올린 젊은이들에게 중형이 | 남 데해지 리에 바 있다.            | 0.0000	False
--------------------------------------------------------------------------------
[15400/100000] Train loss: 0.00003, Valid loss: 0.00462, Elapsed_time: 5553.78908
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
성가족위원회 소속 이자스민 의원은 '일본군 위안부 피해자들에게 법 | 공 롯보호일에 촉앞을 기다자           | 0.0000	False
꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 날정보습 기자                   | 0.0200	False
몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 딱 지 언을 있다.                | 0.0001	False
한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 1 보일에 를리 병곤 기자            | 0.0000	False
정권자는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김 | 입정보입서 이린. 직고 다.           | 0.0000	False
--------------------------------------------------------------------------------
[15500/100000] Train loss: 0.00003, Valid loss: 0.00474, Elapsed_time: 5588.62267
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
주소로돼 있어야 하고 실제 거주해야 한다고 교육청 관계자가 전했 | 딱 해지면 환를 를열된 고보밝다."       | 0.0000	False
한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 을 켜법안 방에 촉각 두있다.          | 0.0000	False
로 고용했으며 로르 회장은 2003년 이래 김 전회장을 최소 세 번 만 | 기북정등보 확산찰이 보원보 니다.        | 0.0000	False
에서 개봉한다. 일본 산이스포츠는 21일 "'마이 브라더('우리형'의 | 빛 총격 해 습 원고 있다.           | 0.0000	False
안심사소위 회의실에는 민노당 의원들이 모두 나와 법안 심사 저지에 | 을 동일 언촉을  기자              | 0.0000	False
--------------------------------------------------------------------------------
[15600/100000] Train loss: 0.00003, Valid loss: 0.00617, Elapsed_time: 5624.09559
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한진중공업 김주익 열사와 두산중공업 배달호 열사 등 수많은 동지들 | 공 켜롯환리안부 을  기자            | 0.0000	False
임직원들로부터 높은 인기를 얻고 있다. 영어, 중국어, 일본어 외에도 | 남북갈등해을 니까 장앞 구속다.         | 0.0000	False
최중경 지식경제부 장관이 경기고 9년 선배인 정운찬 동반성장위원 | 기상정보습 다.                  | 0.0095	False
김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 기 켜보법 산촉을 있다.             | 0.0000	False
니다. 검찰은 유죄 확정 시 범죄수익을 차질 없이 추징하기 위해 재산 | 공 켜보광 를 도 기자              | 0.0000	False
--------------------------------------------------------------------------------
[15700/100000] Train loss: 0.00003, Valid loss: 0.00581, Elapsed_time: 5659.52154
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
각종 모의시험 설비를 갖춘 원자력 서비스센터를 준공했다. 원자력 | 성 게 10 원 니다."             | 0.0000	False
이 가동돼 한은의 통화정책이 영향을 받는 것 아니냐는 우려의 목소 | 공동 롯광 위1에 앞 기자            | 0.0000	False
낮부터 기온이 오름세를 보이면서 추위가 금세 누그러지습니다. 그 | 성마 환를 해니1 0사앞 니다.         | 0.0000	False
차량에 많이 쓰인다. 반면 래그형 은 트럭이나 버스에 장착되는 타이 | 남씨정보  획 있다.               | 0.0006	False
해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 공동 롯데호에 린. 일앞한 있다.        | 0.0000	False
--------------------------------------------------------------------------------
[15800/100000] Train loss: 0.00002, Valid loss: 0.00457, Elapsed_time: 5695.20894
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
서는 등 사태가 갈수록 꼬여가고 있다. 한나라당은 전혀 다른 논리를 | 공 롯호 열린 0만원직 .자           | 0.0000	False
주민의 인권을 실질적으로 증진하고 국제적 기준에 따라 인도적 지원 | 날씨정보 다."                  | 0.0013	False
이 초등학교 친구들보다 1년 늦었다. 아들이 동네 학교에서 친구들의 | 공 랜 호광호 열시린 앞짐저 기.        | 0.0000	False
이 가동돼 한은의 통화정책이 영향을 받는 것 아니냐는 우려의 목소 | 공동롯는 해1에 앞 기자             | 0.0000	False
어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 |  지일에 를을  기자               | 0.0000	False
--------------------------------------------------------------------------------
[15900/100000] Train loss: 0.00004, Valid loss: 0.00543, Elapsed_time: 5730.64520
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
트파일의 틈이 벌어지자 용접하는 데 그다. 또 사고지점 반대편과 | 10조을합면 조짐 바도 기다.          | 0.0000	False
과 유럽 재정위기 등 불안요인이 크고 대내적으로는 물가불안, 가계 | 적 환 니 기전를 습니다."           | 0.0000	False
면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 단 총격을 최세 사앞한 바 있다.        | 0.0000	False
. 타이어는 재질 역시 스노우 타이어의 제동력을 높이는 역할을 하게 | 공 롯보 1에 을 바 있다.           | 0.0000	False
전혀 없었다"고 말했다. 앞서 프랑스 일간 리베라시은 로르 회장의 | 입을보 1광안 를만  기자            | 0.0000	False
--------------------------------------------------------------------------------
[16000/100000] Train loss: 0.00003, Valid loss: 0.00432, Elapsed_time: 5765.57729
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 빛 랜 확산광당 길 있다.            | 0.0000	False
자를 목격한 사실을 확인했다. 경찰은 이씨의 군부대에 이씨의 특별 | 성마비환 를일 기자                | 0.0000	False
체 호워드 리그의 드루 닐슨은 "징역 4년형은 흉기로 상해를 입 | 심로지 를로 를 기자               | 0.0000	False
의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 정보다.                      | 0.4429	False
업부서로 배치해 검사기능과 소비자보호 업무를 대폭 강화한다는 방 | 을 지해보호 y 데방촉를앞을 두있다.      | 0.0000	False
--------------------------------------------------------------------------------
[16100/100000] Train loss: 0.00003, Valid loss: 0.00446, Elapsed_time: 5801.01416
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 기북갈보까지 산 을 바 있다.          | 0.0000	False
다. 조 전 수석은 박 전 대통령이 이 부회장 퇴진을 지시했고 손경식 c | 남 갈등보 난세호 사앞당 있다.         | 0.0000	False
물관리 잘못의 책임을 져야 할 수자원공사 사장이 오히려 연임 것 | 전해  다.                    | 0.0001	False
수 기자                      | 수 기자                      | 0.7336	True
맞는 이 축제는 공음면 학원농장의 보리밭에서 5월 8일까지 열린다. | 딱 총광입세다 경찰저 보이고 있다.       | 0.0000	False
--------------------------------------------------------------------------------
[16200/100000] Train loss: 0.00003, Valid loss: 0.00453, Elapsed_time: 5836.58358
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 빛바 일 일 전드습습니다."           | 0.0000	False
운영할 필요가 있다는 데 공감했다"며 "특히 수출과 내수간 격차, 지 | 입 갈열호면 소열린 앞길 고 있다.       | 0.0000	False
해관리공단은 지난 1월 지식경제부 기술표준원의 12년 국가표준기 | 정보 지2박를니다.                | 0.0000	False
건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 을북보환y처 만앞 있다.             | 0.0000	False
된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 정해 니다.                    | 0.0772	False
--------------------------------------------------------------------------------
[16300/100000] Train loss: 0.00003, Valid loss: 0.00407, Elapsed_time: 5871.59665
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
이다.                       | 이다.                       | 0.6871	True
생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 전해습 있다.                   | 0.0004	False
러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 공 롯데호에서 린 사리직 기자          | 0.0003	False
금실무협의회 이후 십수년만에 부활되는 것이다. 하지만 재정부가 | 입 호에 발를앞 바 있다.            | 0.0000	False
서 받은 자료를 바탕으로 책 두권 분량의 탄원서를 만들어 금감독 | 한다 구지최 최세호 장병호 다자         | 0.0000	False
--------------------------------------------------------------------------------
[16400/100000] Train loss: 0.00003, Valid loss: 0.00457, Elapsed_time: 5907.09759
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
흡수된다. 리스크검사지원국은 폐지된다. 금감원은 또 소비자서비스 | 날씨보앞 다.                   | 0.0009	False
찾는 이들이 많다. 하지만 여름 보양식에 와인을 곁들이면 더욱 좋다 | 공 롯호 환 조세린 앞입니다."         | 0.0000	False
목할 것으로 검찰은 보고 있다. 검찰은 현 전 의원에 대한 전방위 자금 | 정보습니다.                    | 0.1696	False
군산시에서는 5월 4일부터 8일까지 5월의 보리밭, 추억속으로 안내 | 공 환광위 니1다 0원 바니다.         | 0.0000	False
그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 소문는 지텔의다 를언앞 앞원바니다."      | 0.0000	False
--------------------------------------------------------------------------------
[16500/100000] Train loss: 0.00002, Valid loss: 0.00508, Elapsed_time: 5942.64655
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 공동조을열면 소린다 촉을 .자          | 0.0000	False
이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 딱보해세 박사를 앞당 계보있다.         | 0.0000	False
기 정보를 확인하야습니다. 오늘 낮 최고 기온은 서울 2도 등 어 | 1 롯데텔법에 촉을 바 기자           | 0.0000	False
두확인한 결과, 김우중 전 회장이 출국한 이후 어느 쪽 여권으로도 귀 | 남 보난지 조이 보 있다.            | 0.0000	False
입니다.                      | 입이다.                      | 0.5447	False
--------------------------------------------------------------------------------
[16600/100000] Train loss: 0.00003, Valid loss: 0.00411, Elapsed_time: 5977.54379
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
지하철 7호선 방화사건을 수사중인 경기도 광명경찰서는 사건당일 | 을동보지1산 짐n 바 다.자           | 0.0000	False
된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 정해 니다.                    | 0.0933	False
태가 많이 쓰이기도 한다. 반면 스노우 타이어 트레드는 깊이가 깊고 | 빛 보호 세 y이 보 기다.           | 0.0000	False
을 감안해 집행유예를 선고한다"고 밝다. 노씨는 2003년 10월께 | 을랜광입니린 사직 기자              | 0.0000	False
이는 방식을 택해 공장식 축산을 할 수 없다며 한정생산을 강점으로 | 1 해지 를을 저 앞정전 있다.         | 0.0000	False
--------------------------------------------------------------------------------
[16700/100000] Train loss: 0.00003, Valid loss: 0.00449, Elapsed_time: 6013.11985
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
성은 배제할 수 없을것으로 보여 향후 검찰의 수사가 주목된다. | 빛마 영광입니 앞당길고 있다.          | 0.0000	False
파클을 함께 마시면 입안에서 청량감이 살아나 더운 날씨에 처진 | 남북보1y 앞원  이다.             | 0.0000	False
고 있습니다. 화재 예방에 유의하야습니다. 내일은 전국에 구름 |  앞 기자                     | 0.0050	False
김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 남북켜보법안 산전해을드습 다.          | 0.0000	False
글로스 주식 77억원 어치를 매수하는 과정에서 박 씨와 짜고 주가 | 정보 기자                     | 0.0380	False
--------------------------------------------------------------------------------
[16800/100000] Train loss: 0.00002, Valid loss: 0.00421, Elapsed_time: 6048.64661
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
국이 신설되고, 일부 실이 국으로 승격돼 기능이 강화된다. 13일 금 | 기정보습 니다."                 | 0.0102	False
몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 을 지켜며법안 방에 촉각을 곤 있다.      | 0.0029	False
이 과제가 국가 추진과제로 선정다. 이에 따라 오는 2014년을 목 | 빛 환여등 세면 경이 다.            | 0.0000	False
위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 빛바 일 일 전드습니다."            | 0.0000	False
로 행복 가치를 추구해 기 때문"이라고 강조했다. 1937년 경성우 | 단과총격 합최 경하이 도 있다.         | 0.0000	False
--------------------------------------------------------------------------------
[16900/100000] Train loss: 0.00003, Valid loss: 0.00469, Elapsed_time: 6083.85309
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
환으로, 바른 역사관과 정체성을 가지고 사회적 책임을 다하는 기업 | 남0격해지호 조열1면 0원니다.         | 0.0000	False
사용할 계획이다. 이재우 사장은 "이번 협약 체결로 병마와 싸우고 | 입갈입열 열하 저계했다.             | 0.0000	False
운영할 필요가 있다는 데 공감했다"며 "특히 수출과 내수간 격차, 지 | 입 열호면 소열린 앞길 고 있다.        | 0.0000	False
으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 빛 영보광 광을  기자              | 0.0000	False
서의 역할이라고 밝히고 있다. 검찰은 사실관계를 확인하기 위해 박 | 남북정보  계병 기자               | 0.0000	False
--------------------------------------------------------------------------------
[17000/100000] Train loss: 0.00003, Valid loss: 0.00461, Elapsed_time: 6119.29019
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
고 거듭 밝다. 김형곤의 동생이 산 암태면 벌목도는 신안군내 753 | 기북정보습이 사정보습니다."           | 0.0001	False
과 유럽 재정위기 등 불안요인이 크고 대내적으로는 물가불안, 가계 | 적 환 을니 기전를습니다."           | 0.0000	False
입장을 발표하고, 사회적으로 손가락질받을 일이 학교에서 발생해 안 | 남북갈등 산보갖 니다.              | 0.0001	False
한진중공업 김주익 열사와 두산중공업 배달호 열사 등 수많은 동지들 | 공 북리안 를을  기자              | 0.0000	False
앙은행 독립성 손 논란이 있는 가운데 양 기관간 차관급 대화채널 | 북갈보습산 고도 있기.              | 0.0000	False
--------------------------------------------------------------------------------
[17100/100000] Train loss: 0.00002, Valid loss: 0.00454, Elapsed_time: 6154.18409
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
양구경찰서에 육군 모 사단에서 복무중인 사병 이모(22)씨가 112전 | 남북정보다.                    | 0.0159	False
해주지 않으면 주가조작 사실을 금감독원에 신고하다"고 협박한 | 딱보  기자                    | 0.0068	False
다. 서울우유는 지난 11일 75주년 기념식에서 '행복한 젖소, 행복한 | 1 기자                      | 0.0751	False
적 코리코 303호(860t급)의 선원 7명 가운데 이날 오후 7시 40분께 | 성 지환를 을1 0조원입니다."         | 0.0000	False
전입학시다는 일부 언론의 보도는사실무근이다"고 해명한 것으로 | 성다"  있다자                  | 0.0003	False
--------------------------------------------------------------------------------
[17200/100000] Train loss: 0.00003, Valid loss: 0.00454, Elapsed_time: 6189.87857
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
의했기 때문에 필요할 경우 c검사는 참고인 자격으로 소환 가능 | 와 영광등을  기자                | 0.0000	False
건축학과를 졸업하고 현재중공업분야에서 해외 수출 업무를 전담하 | 남 갈등으 산저 했다자              | 0.0000	False
인터넷을 차단했던 이집트 정부처럼 행동하려 한다고 비난했다.  | 딱 롯해호에 린산 저 있다.           | 0.0000	False
500만원이 아 3억원을 받은 의로 구속되면 더 이상 금품수수 사 | 빛 동 일광 촉등을  고 있다.         | 0.0000	False
리고 모레 밤에는 경기 북부와 강원 영서 북부에 비가 내리습니다. | 기 갈등광 산열1 찰이 고 이다.        | 0.0000	False
--------------------------------------------------------------------------------
[17300/100000] Train loss: 0.00002, Valid loss: 0.00447, Elapsed_time: 6225.23813
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
과 다르게 경영실적보고서를 작성해 높은 등급을 받다. 이 덕분에 | 와 지일 촉각을 다.               | 0.0000	False
부와 한은 간에 자료협조, 경제상황에 대한 의견교환 등 보다 긴밀한 | 날씨정보 원고 기자                | 0.0000	False
를 위해 2학기부터 수업을 분리, 운영할 계획이라고 했다. 이와 함께 | 와 동일부 을  기자               | 0.0000	False
고생할 것으로 우려해 딸에게 극약을 먹여 숨지게 한 의로 구속기 | 북정보산 을 보종 기자              | 0.0000	False
러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 공 롯데호에서 린 사리직 기자          | 0.0003	False
--------------------------------------------------------------------------------
[17400/100000] Train loss: 0.00003, Valid loss: 0.00465, Elapsed_time: 6260.16447
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
해주지 않으면 주가조작 사실을 금감독원에 신고하다"고 협박한 | 귀 기자                      | 0.1074	False
반성이 있어야 한다"고 말했다. 이에 대해 문 후보측 김경수 공보특보 | 남정보 산마 앞사앞곤 기자            | 0.0000	False
한나라당의 '역공'으로 풀이된다. 김 지사측은 한나라당 차원의 의도 | 날 상북해보습니세 0만을 곤 있다.       | 0.0000	False
의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 날북정으보 니세 저고 다.            | 0.0000	False
목할 것으로 검찰은 보고 있다. 검찰은 현 전 의원에 대한 전방위 자금 | 정보습이다.                    | 0.1021	False
--------------------------------------------------------------------------------
[17500/100000] Train loss: 0.00002, Valid loss: 0.00455, Elapsed_time: 6295.64754
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
외은지점감독실이 신설된다. 중소서민금 권역에서는 여신전문서 | 딱광 를각  기자                 | 0.0000	False
뒤 취한 조니다. 법원에 청구한 동결 대상 재산은 28억 원에 매입한 | 다.                        | 0.5297	False
리고 모레 밤에는 경기 북부와 강원 영서 북부에 비가 내리습니다. | 기 갈등광 산열1 찰이 고 이다.        | 0.0000	False
은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 빛바 영일 계고 있다.              | 0.0000	False
식이 열리는 저녁에는 체감온도가 영하 10도 안팎으로 떨어질 것으 | 남 갈등으보해 산세 저 장전고 다자       | 0.0000	False
--------------------------------------------------------------------------------
[17600/100000] Train loss: 0.00003, Valid loss: 0.00416, Elapsed_time: 6331.25723
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 와 여등을 경 했다.               | 0.0000	False
채용할 경우 증원인력 1인당 월 50만원(올해 60만원)씩한시적으로 | 남정보습니까 산 고 있다.            | 0.0000	False
28일 독립영화의 산실인 전주국제영화제가 화려한 막을 올린다. 올 | 남 켜보입니 열찰린 사망기도 있다.       | 0.0000	False
개월 만에 불명예 퇴진했습니다.본선행에 빨간 불이 켜진 대표팀을 | 남갈보합산 조짐마 사하기고 기다.        | 0.0000	False
고 더 큰 사회 문제를 일으키는 원인이  것"이라며 "결국 그들이 당 | 빛마 영광을 경저이보호 기자           | 0.0000	False
--------------------------------------------------------------------------------
[17700/100000] Train loss: 0.00002, Valid loss: 0.00434, Elapsed_time: 6366.12496
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 날로씨정보전 다.                 | 0.0000	False
화축제가 전주시 풍남동 경기전 일대에서 열린다. 임실군에선 30일 | 빛마랜 랜광위 1경찰 0만원 다.        | 0.0000	False
다. 하지만 민주노총은 아직 마음의 문을 열지 않다. 참여정부 노동 | 한다  있다.                   | 0.0001	False
로 승승장구했던 대표팀,최종 예선전 성적표는 참혹했습니다.지난해 | 와 동일여 을y 앞원 기자            | 0.0000	False
톱스타 원이 오는 27일 일본 요코하마에서 자선 팬미팅을 연다. 원 | 날 기다자                     | 0.0527	False
--------------------------------------------------------------------------------
[17800/100000] Train loss: 0.00002, Valid loss: 0.00469, Elapsed_time: 6401.61154
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
다. 서울우유는 지난 11일 75주년 기념식에서 '행복한 젖소, 행복한 | 남0갈을까지 산 조짐저 보이다 기다.      | 0.0000	False
로 했던 민주노총은 오후 11시 투쟁본부회의를 열어 총파업 방침에 | 단다 치 한 바 있다.              | 0.0000	False
오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 성 일게 해니다.                 | 0.0019	False
딱딱해지는 특성을 갖고 있다.          | 딱딱해지는 특등을 갖고 있다.          | 0.0077	False
정책 우선순위를 두는 가운데 고용회복이 지속 수 있는 방향으로 | 성마 총격을 한  있다자             | 0.0000	False
--------------------------------------------------------------------------------
[17900/100000] Train loss: 0.00003, Valid loss: 0.00462, Elapsed_time: 6437.22297
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
후배가 되는 것을 싫어하는 것을 보고 아내의 지인이 사는 강동구로 | 한 기자                      | 0.3482	False
자원공사측에 건의했다. 하지만 수자원공사는 사고직전 가물막이 시 | 입 롯광입면y를 앞 있다.            | 0.0000	False
한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 남씨정보 있다.                  | 0.0051	False
족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 을 보광안 를앞당  다.             | 0.0000	False
권에 따르면 금감원은 이같은 내용의 조직개편안을 마련하고 조만간 | 대로으보 장보병호 기자              | 0.0000	False
--------------------------------------------------------------------------------
[18000/100000] Train loss: 0.00003, Valid loss: 0.00405, Elapsed_time: 6472.43171
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
. 내일은 오늘보다 기온이 더 높습니다. 서울 아침 기온 영하 6도, | 남북원을합지 확산조된 사망하 고 있다.     | 0.0000	False
하는 점에 비 b고교의 전입학 시스템에 중대 허점이 있었던 사실도 | 전보문 확 짐마저 고 있다.           | 0.0000	False
송수진입니다."                  | 송수진입니다."                  | 0.4715	True
역으로 지목하고 있는 '다이아드 아일랜드'에 포함돼 있어 이 같은 | 딱해보 를했다.                  | 0.0000	False
트파일의 틈이 벌어지자 용접하는 데 그다. 또 사고지점 반대편과 | 1 롯데텔에 앞을 기자              | 0.0000	False
--------------------------------------------------------------------------------
[18100/100000] Train loss: 0.00002, Valid loss: 0.00413, Elapsed_time: 6508.01493
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
와의 동일여부 등을 확인할 예정이다.      | 와의 동일여부 등을 확인할 예정이다.      | 0.0995	True
권 수익 전액을 일본 유니세프에 기부하기로 했다. 원의 매니저 장 | 단과 격을 입면 앞당 있다.           | 0.0000	False
의로 기소된 노모(38.여)씨에 대해 징역 3년에 집행유예 5년을 선 | 날씨으보 0사도 기자               | 0.0000	False
로 삭감는데도 정파적으로 몰고가는 것은 아집과 독선"이라고 말했 | 기씨보 보습 기자                 | 0.0000	False
단축에 따른 중소기업의 부담을 덜어주기 위해지원사업을 펴고 있으 | 공 동보여 전당저 저보정 이다.         | 0.0000	False
--------------------------------------------------------------------------------
[18200/100000] Train loss: 0.00002, Valid loss: 0.00451, Elapsed_time: 6542.95831
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
올린 것에 대해서는 지나친 중형"이라고 주장했다. 반면 공공의 안녕 | 빛문이으구 세호 사장병호 있다.         | 0.0000	False
블록패을 띄고 있다. 자동차가 눈길을 주행할 때 바퀴가 눈을 누르 | 남북갈등으로 산전앞저고 기자           | 0.0000	False
성은 배제할 수 없을것으로 보여 향후 검찰의 수사가 주목된다. | 빛바랜 영광입니 앞당길고 있다.         | 0.0000	False
크 서비스를 규제하는 방안을 검토하다고 밝다. 데일리직 길인 | 딱 해광 을 을세다 저이고보 있다.       | 0.0000	False
둥 번개가 치습니다. 비는 밤에 서쪽 지역부터 차차 그치습니다. | 전보제 있다.                   | 0.0002	False
--------------------------------------------------------------------------------
[18300/100000] Train loss: 0.00002, Valid loss: 0.00461, Elapsed_time: 6578.47167
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
통해 금감원 검사에서 '강도와 제재수준을 완화해달라'는 취지로 당 | 을 보지면 시찰를 앞길 기자           | 0.0000	False
교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 적 롯광일해 해정있다.              | 0.0000	False
이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 입을보로합지면 소박를 속세다.          | 0.0000	False
을 챙긴 의(자본시장통합법 위반) 등으로 국제금브로커 이 모씨 | 을 켜법 처향방에 촉각을 곤두있다.       | 0.0000	False
들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 입 지보열 열세린 저 보하기도 있다.      | 0.0000	False
--------------------------------------------------------------------------------
[18400/100000] Train loss: 0.00003, Valid loss: 0.00484, Elapsed_time: 6614.03512
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 을 지환열리에 처리를 를을 기자         | 0.0000	False
한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 적보리 이다.                   | 0.0001	False
입니다.                      | 입이다.                      | 0.5676	False
크 서비스를 규제하는 방안을 검토하다고 밝다. 데일리직 길인 | 딱해드습당 앞보습 있다자             | 0.0000	False
을 감안해 집행유예를 선고한다"고 밝다. 노씨는 2003년 10월께 | 을랜광입니 y 사직 기자             | 0.0000	False
--------------------------------------------------------------------------------
[18500/100000] Train loss: 0.00002, Valid loss: 0.00460, Elapsed_time: 6648.96949
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 남정보  했다.                  | 0.0067	False
하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 다 지취는 를을 한  있기자           | 0.0000	False
에 오른다. 이 기간 동안 방단은 블라디보스 한인 집단 거주지, 안 | 소문이지 확산를 있다."             | 0.0000	False
이들 중 15학번 남학생 9명은 주점에 후배 남학생들을 불러 동료 여 | 을 롯호서 린 촉오곤 기자            | 0.0000	False
고 거듭 밝다. 김형곤의 동생이 산 암태면 벌목도는 신안군내 753 | 기 북갈등보 1 0조원곤 기자          | 0.0000	False
--------------------------------------------------------------------------------
[18600/100000] Train loss: 0.00002, Valid loss: 0.00429, Elapsed_time: 6684.46865
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
여학생 중에 (성관계를) 하고 싶은 사람을 골라라는 말을 한 것으로 | 적 동 방에 언을 다.              | 0.0000	False
시키기 위해 위장전입한 것은 사실이나 오씨의 도움을 받아 b고교에 | 1 지켜보만  기.자               | 0.0000	False
인 피고인이 잘못을 뉘우치며 정신적 후유증에 시달리고 있는 점 등 | 성마 총보격위경를했 기다자            | 0.0000	False
소재로 한 '2011 고창 복분자 푸드페스티벌'을 개최해 축제의 계절에 | 한 지 발 앞한 한 기자             | 0.0000	False
우유, 행복한 고객'을 실현하다고 선언했다. 75년을 넘어 100년으 | 적 높 입니다 사를 앞전입니다.         | 0.0000	False
--------------------------------------------------------------------------------
[18700/100000] Train loss: 0.00002, Valid loss: 0.00461, Elapsed_time: 6720.00911
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
고 있습니다. 화재 예방에 유의하야습니다. 내일은 전국에 구름 | 을로보로 지산입 앞고 기자            | 0.0000	False
고 있다. 야당과 시민사회단체들은 중형을 선고한 것은 지나치다고 | 한의 지 를을 니다.               | 0.0000	False
화를 추진하다고 밝 습니다. 평창 올림픽에 대해서는 88 올림픽 | 입보안 광만당 앞 기자              | 0.0000	False
한진중공업 김주익 열사와 두산중공업 배달호 열사 등 수많은 동지들 | 공 북리안부 를을  기자             | 0.0000	False
국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 남북갈보호면환0 촉각원 있다.          | 0.0000	False
--------------------------------------------------------------------------------
[18800/100000] Train loss: 0.00002, Valid loss: 0.00447, Elapsed_time: 6754.88971
Current_accuracy : 3.700, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
금감원은 은행과 중소서민 검사 담당 부원장보 자리를 신설해 일반은 | 입문 지텔에 촉방앞 앞을곤두있다.        | 0.0000	False
와의 동일여부 등을 확인할 예정이다.      | 와의 동일여부 등을 확인할 예정이다.      | 0.1083	True
9월 중국전 한 골 차 진땀승을 시작으로, 조 최약체로 꼽히던 시리아 | 의을문 확광확를당된다 고보했다.         | 0.0000	False
귀국치 않다.                   | 귀국치 않다.                   | 0.7694	True
국으로 확대 개편하는 등 조직을 대폭 확대한다. 특히 감독과 검사업 | 딱 지는 등을 확저 보호 기자          | 0.0002	False
--------------------------------------------------------------------------------
[18900/100000] Train loss: 0.00003, Valid loss: 0.00436, Elapsed_time: 6790.51420
Current_accuracy : 3.700, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
이 곧 글로벌 기업이  수 있다는 의지가 담겨있다고 회사측은 설명 | 남갈등까 짐마저 보이보있다."          | 0.0000	False
최중경 지식경제부 장관이 경기고 9년 선배인 정운찬 동반성장위원 | 기상정보습 다.                  | 0.0081	False
불 축제'를 연다.                | 불 축제'를 연다.                | 0.3917	True
고 있는 추세라는 것이 회사 측 설명이다. 사이버 강의는 스마트으 | 을 는으 지 확 산당 기자            | 0.0000	False
는 이번 방은 안중근 의사 관련 레포트 우수 대학생 13명, 전국 중 | 남북정보습 산조을 바 다."           | 0.0000	False
--------------------------------------------------------------------------------
[19000/100000] Train loss: 0.00002, Valid loss: 0.00493, Elapsed_time: 6826.08184
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
년 가까이 끌어온 북한인권법을 이번 임시국회에서 반드시 처리하 | 성환광  기자                   | 0.0000	False
제협력국이 기획조정국, 총무국, 공보실과 함께 기획총 부문으로 | 소문 지확일를 습된 앞원습니다.         | 0.0000	False
대안으로 마련한 것으로 4대강사업을 반대해 온 김두관 지사에 대한 | 대피해이다.                    | 0.0170	False
밝다. 박씨도 로비 자체를 부인하고 있다. 박씨는 부산저축은행에 | 날 기자                      | 0.0415	False
피해자 문제 제기를 하는 등 보다 적극적인 자세를 보여야 한다"고 말 | 성마 환를광등을  산 기자            | 0.0000	False
--------------------------------------------------------------------------------
[19100/100000] Train loss: 0.00003, Valid loss: 0.00503, Elapsed_time: 6861.03194
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
입니다.                      | 입이다.                      | 0.5598	False
조 2위로 힘겹게 이뤄 '9회 연속 월드컵 본선행'.상처로 가득했던 | 기상정보 전보습 원니다"             | 0.0000	False
한국노총도 법안 강행시 벌이기로 했던 노사정위 탈퇴와 대정부 투쟁 | 남북등으로까지 확산조짐 저 기자         | 0.0000	False
으로 알려지고 있다. 지난 5월 8일 구미시와 국가공단의 생명줄인 해 | 을면 t만 기자                  | 0.0000	False
맞는 이 축제는 공음면 학원농장의 보리밭에서 5월 8일까지 열린다. | 전총광입세다 경찰저 보이고 있다.        | 0.0000	False
--------------------------------------------------------------------------------
[19200/100000] Train loss: 0.00004, Valid loss: 0.00475, Elapsed_time: 6896.62165
Current_accuracy : 3.900, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
된다. 고무에 카본블주1을 섞은 일반타이어는 날씨가 추워질수록 | 빛 지환광 등 당 저고 있다.          | 0.0000	False
주 고등법원 순국장소인 순 감옥 등 중국, 러시아의 안중근 의사가 | 딱딱해지 해드 있다.               | 0.0001	False
조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 을 동일법 언을 한 보있다.           | 0.0000	False
그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 심을로 지세다 데리서 앞길오종호 기자      | 0.0000	False
러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 공 롯데호에서 린 사리 기자           | 0.0003	False
--------------------------------------------------------------------------------
[19300/100000] Train loss: 0.00003, Valid loss: 0.00465, Elapsed_time: 6931.59086
Current_accuracy : 3.800, Current_norm_ED  : 0.16
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
데 인식을 같이했다"고 밝다. 양 기관은 또 "거시정책은 물가안정에 | 입 입다 찰이 앞 고 있다.           | 0.0000	False
사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 남북갈보열지 환0 방에 곤호 다.        | 0.0000	False
지급률 수정 등 적절한 조치를 취하라"고 통보하고, 수공 사장에게 " | 딱로지해를 를당n 만를 있다.          | 0.0000	False
자들의 명예가 손 우려가 있는 사건들이 늘고 있다"며 "일본군 위 | 소문이 조짐1 촉앞을한 기자           | 0.0000	False
양구경찰서에 육군 모 사단에서 복무중인 사병 이모(22)씨가 112전 | 입정보 기자                    | 0.0513	False
--------------------------------------------------------------------------------
[19400/100000] Train loss: 0.00003, Valid loss: 0.00430, Elapsed_time: 6967.21754
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 입보호서열린다 장일병 다.            | 0.0000	False
족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 단 총격을 지난산 앞저 보호 다.        | 0.0000	False
입니다. 다만 제주도는 일요일 오후부터 다시 비가 오기 시작하습 | 빛바" 해 속입저 망도 있다.          | 0.0000	False
사에게 넘겨준것으로 알고 있다. 원래 학생과 학부모는 함께 전입학 |  다.                       | 0.0174	False
사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 을 보호지서 0기 촉각을곤입니다.        | 0.0000	False
--------------------------------------------------------------------------------
[19500/100000] Train loss: 0.00002, Valid loss: 0.00410, Elapsed_time: 7002.84711
Current_accuracy : 3.900, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
원장, 감사원장에게 보 사실은 인정하고 있으며 이는 고문변호사로 | 빛 동보일  기다자                | 0.0000	False
다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 기보광  다.                   | 0.0000	False
생활고에 시달리다 남편과 동반자살을 하기로 한 뒤 혼자남는 딸이 | 성랜광입 산세마 길 고 니다."         | 0.0000	False
호사는 박 전 대통령의 현금 10억여 원도 보관 중입니다. 지금까지 드 | 날씨정보  기자                  | 0.0126	False
이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 1 보해최세 사조앞를 앞 보있다.        | 0.0000	False
--------------------------------------------------------------------------------
[19600/100000] Train loss: 0.00002, Valid loss: 0.00421, Elapsed_time: 7037.77328
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
자를 목격한 사실을 확인했다. 경찰은 이씨의 군부대에 이씨의 특별 | 성바비환 광일 기자                | 0.0000	False
한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 을 지켜며안 을안 곤두세다.           | 0.0000	False
된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 기 정으 보로 지저 전니이다.          | 0.0000	False
위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 빛바 일 일 전보습습니다."           | 0.0000	False
표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 빛 지랜 처광당 길 있다.            | 0.0000	False
--------------------------------------------------------------------------------
[19700/100000] Train loss: 0.00002, Valid loss: 0.00411, Elapsed_time: 7073.25448
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 3.900, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
생활고에 시달리다 남편과 동반자살을 하기로 한 뒤 혼자남는 딸이 | 남랜광입 입세마 길 고 니다."         | 0.0000	False
이다. 소음이 많은 것이 단점으로 꼽히기는 하지만 전후 방향의 강력 | 남북갈등으보까 산조앞고 보 다자         | 0.0000	False
금감독원이 금서비스개선국을 신설하고 외환업무실을 외환감독 | 빛 랜광에 를을 y 사앞 다.          | 0.0000	False
동 자택과 1억 원짜리 수표 30장 등 모두 58억 원 규몹니다. cj그룹 | 공 보광입니린 촉으를  다기자          | 0.0000	False
법무부 공보관은 "김우중씨 명의로 된 한국 여권과 프랑스 여권을 모 | 공총보 지언 를 기자               | 0.0000	False
--------------------------------------------------------------------------------
[19800/100000] Train loss: 0.00002, Valid loss: 0.00382, Elapsed_time: 7108.77601
Current_accuracy : 4.000, Current_norm_ED  : 0.15
Best_accuracy    : 4.000, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
로 삭감는데도 정파적으로 몰고가는 것은 아집과 독선"이라고 말했 | 날정보 기자                    | 0.0231	False
한진중공업 김주익 열사와 두산중공업 배달호 열사 등 수많은 동지들 | 공 롯일리에 를을  기자             | 0.0000	False
은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 을 지켜며리안 1 앞을곤 기자          | 0.0000	False
조합 조합장은 "서울우유가 75년간 업계 1위 자리를 지켜올 수 있었 | 을문 이다 구최호기장보습니다."         | 0.0000	False
국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 을 지보광호 서 시린 촉각을호 기자       | 0.0000	False
--------------------------------------------------------------------------------
[19900/100000] Train loss: 0.00003, Valid loss: 0.00411, Elapsed_time: 7144.31912
Current_accuracy : 3.800, Current_norm_ED  : 0.15
Best_accuracy    : 4.000, Best_norm_ED     : 0.17
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
입을 열면 소환시기를 앞당길 계획이다.     | 입을 열면 소환시기를 앞당길 계획이다.     | 0.5104	True
군산시에서는 5월 4일부터 8일까지 5월의 보리밭, 추억속으로 안내 | 공 환광 1다 조원을곤바니다.          | 0.0000	False
된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 정보 기자                     | 0.0475	False
놔두기 어렵다는 점과 3억원이 조씨에게 전달다는 여러 증거가 드 | 단 격 세1 0원을 호 기자           | 0.0000	False
국노동연구원 한 연구위원은 "참여정부가 신자유주의라는 흐름에서 | 남북갈보까 등 한 바 기자            | 0.0000	False
--------------------------------------------------------------------------------
Traceback (most recent call last):
  File "train.py", line 317, in <module>
    train(opt)
  File "train.py", line 148, in train
    image = image_tensors.to(device)
KeyboardInterrupt
^C
```

```
!python3 train.py \
    --train_data ../data_aihub/lmdb/train \
    --valid_data ../data_aihub/lmdb/valid \
    --Transformation TPS \
    --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM \
    --Prediction CTC \
    --num_iter 100000 \
    --valInterval 100 \
    --batch_max_length 45 \
    --batch_size 64 \
    --data_filtering_off
```

## 학습 3: TPS-VGG-BiLSTM-Attn

- 학습결과분석
    - 약 6시간 20분 학습
    - Trainable params num : 12086234
    - attention 모델이 상당히 효과적임 약 84%의 정답률 달성
    - 최고 정답률 84%
    
    ![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20167b04eba1464a9da95f0f099b6bf9e0/Untitled%203.png)
    
- 학습 결과
    
    ```go
    --------------------------------------------------------------------------------
    dataset_root: ../data_aihub/lmdb/train
    opt.select_data: ['/']
    opt.batch_ratio: ['1']
    --------------------------------------------------------------------------------
    dataset_root:    ../data_aihub/lmdb/train	 dataset: /
    sub-directory:	/.	 num samples: 10000
    num total samples of /: 10000 x 1.0 (total_data_usage_ratio) = 10000
    num samples of / per batch: 128 x 1.0 (batch_ratio) = 128
    --------------------------------------------------------------------------------
    Total_batch_size: 128 = 128
    --------------------------------------------------------------------------------
    dataset_root:    ../data_aihub/lmdb/valid	 dataset: /
    sub-directory:	/.	 num samples: 1000
    --------------------------------------------------------------------------------
    model input parameters 32 100 20 1 512 256 1010 45 TPS VGG BiLSTM Attn
    Skip Transformation.LocalizationNetwork.localization_fc2.weight as it is already initialized
    Skip Transformation.LocalizationNetwork.localization_fc2.bias as it is already initialized
    Model:
    DataParallel(
      (module): Model(
        (Transformation): TPS_SpatialTransformerNetwork(
          (LocalizationNetwork): LocalizationNetwork(
            (conv): Sequential(
              (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
              (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (6): ReLU(inplace=True)
              (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (10): ReLU(inplace=True)
              (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (12): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (13): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (14): ReLU(inplace=True)
              (15): AdaptiveAvgPool2d(output_size=1)
            )
            (localization_fc1): Sequential(
              (0): Linear(in_features=512, out_features=256, bias=True)
              (1): ReLU(inplace=True)
            )
            (localization_fc2): Linear(in_features=256, out_features=40, bias=True)
          )
          (GridGenerator): GridGenerator()
        )
        (FeatureExtraction): VGG_FeatureExtractor(
          (ConvNet): Sequential(
            (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): ReLU(inplace=True)
            (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (7): ReLU(inplace=True)
            (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (9): ReLU(inplace=True)
            (10): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
            (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (12): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (13): ReLU(inplace=True)
            (14): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (15): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (16): ReLU(inplace=True)
            (17): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
            (18): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))
            (19): ReLU(inplace=True)
          )
        )
        (AdaptiveAvgPool): AdaptiveAvgPool2d(output_size=(None, 1))
        (SequenceModeling): Sequential(
          (0): BidirectionalLSTM(
            (rnn): LSTM(512, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
          (1): BidirectionalLSTM(
            (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
        )
        (Prediction): Attention(
          (attention_cell): AttentionCell(
            (i2h): Linear(in_features=256, out_features=256, bias=False)
            (h2h): Linear(in_features=256, out_features=256, bias=True)
            (score): Linear(in_features=256, out_features=1, bias=False)
            (rnn): LSTMCell(1266, 256)
          )
          (generator): Linear(in_features=256, out_features=1010, bias=True)
        )
      )
    )
    Trainable params num :  12086234
    Optimizer:
    Adadelta (
    Parameter Group 0
        eps: 1e-08
        foreach: None
        lr: 1
        maximize: False
        rho: 0.95
        weight_decay: 0
    )
    ------------ Options -------------
    exp_name: TPS-VGG-BiLSTM-Attn-Seed1111
    train_data: ../data_aihub/lmdb/train
    valid_data: ../data_aihub/lmdb/valid
    manualSeed: 1111
    workers: 4
    batch_size: 128
    num_iter: 100000
    valInterval: 500
    saved_model: 
    FT: False
    adam: False
    lr: 1
    beta1: 0.9
    rho: 0.95
    eps: 1e-08
    grad_clip: 5
    baiduCTC: False
    select_data: ['/']
    batch_ratio: ['1']
    total_data_usage_ratio: 1.0
    batch_max_length: 45
    imgH: 32
    imgW: 100
    rgb: False
    character:  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률르른를름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없었엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄했행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘
    sensitive: False
    PAD: False
    data_filtering_off: True
    Transformation: TPS
    FeatureExtraction: VGG
    SequenceModeling: BiLSTM
    Prediction: Attn
    num_fiducial: 20
    input_channel: 1
    output_channel: 512
    hidden_size: 256
    num_gpu: 1
    num_class: 1010
    ---------------------------------------
    
    [1/100000] Train loss: 6.90783, Valid loss: 6.69653, Elapsed_time: 3.20413
    Current_accuracy : 0.000, Current_norm_ED  : 0.17
    Best_accuracy    : 0.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 |                                               | 0.0000	False
    고 있습니다. 화재 예방에 유의하야습니다. 내일은 전국에 구름 |                                               | 0.0000	False
    청구했다. 잦은 말바꾸기로 증거인멸 등의 우려가 있는 조씨를 그냥 |                                               | 0.0000	False
    부와 한은 간에 자료협조, 경제상황에 대한 의견교환 등 보다 긴밀한 |                                               | 0.0000	False
    한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 |                                               | 0.0000	False
    --------------------------------------------------------------------------------
    [500/100000] Train loss: 4.78541, Valid loss: 4.89907, Elapsed_time: 201.88378
    Current_accuracy : 0.000, Current_norm_ED  : 0.17
    Best_accuracy    : 0.000, Best_norm_ED     : 0.17
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    신선함을 생명으로 하는 흰 우유의 새로운 기준을 제시했다. 젖소도 | 다                                             | 0.0000	False
    성가족위원회 소속 이자스민 의원은 '일본군 위안부 피해자들에게 법 | 다                                             | 0.0000	False
    은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 다                                             | 0.0000	False
    한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 다                                             | 0.0000	False
    리고 모레 밤에는 경기 북부와 강원 영서 북부에 비가 내리습니다. | 다                                             | 0.0000	False
    --------------------------------------------------------------------------------
    [1000/100000] Train loss: 4.38677, Valid loss: 5.26972, Elapsed_time: 397.40455
    Current_accuracy : 0.000, Current_norm_ED  : 0.23
    Best_accuracy    : 0.000, Best_norm_ED     : 0.23
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    금감독원이 금서비스개선국을 신설하고 외환업무실을 외환감독 | 다. 있다. 이 있다. 1 있다. 1 있다. 있다. 있 | 0.0000	False
    은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 다. 이 있다. 이 대 한 있다. 이 한 있다. 이 | 0.0000	False
    m의 눈이 내리습니다. 내일 비가 오는 지역에서는 돌풍과 함께 천 | 다. 있다. 이 있다. 1 한 있다. 있다. 있다. 있 | 0.0000	False
    국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 다. 이 있다. 이 대 한 있다. 이 한 있다. 이 | 0.0000	False
    21일 오후 5시 10분께 충북 영동군 양산면 가곡리 이모(57.농업)씨 | 다. 있다. 이 있다. 이 있다. 이 있다. 있 | 0.0000	False
    --------------------------------------------------------------------------------
    [1500/100000] Train loss: 4.12302, Valid loss: 5.55755, Elapsed_time: 590.67026
    Current_accuracy : 0.100, Current_norm_ED  : 0.20
    Best_accuracy    : 0.100, Best_norm_ED     : 0.23
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    보했지만 c검사는 "말미를 달라"며 출석을 미뤄온 것으로 알려다. | 이 있다. 이 있다. 이 있다. 이 있다. 이 있다. 것 | 0.0000	False
    에 참석한 민주노총 한 핵심관계자는 "참여정부에서 가장 실패한 정 | 한 있다. 이 이 이 이 이 이 이 이 이 이 이 이 이 이 | 0.0000	False
    각종 모의시험 설비를 갖춘 원자력 서비스센터를 준공했다. 원자력 | 로 있다. 이 있다. 이 있다. 이 있다. 이 있다. 이 있 | 0.0000	False
    대한 재검토와 향후대책을 논의하는 등 비상 대기상태에 들어다. | 한 이 있다. 이 전 전 이 있다. 이 전 전 있다. 이 | 0.0000	False
    한은 금통화위원회에 차관을 참석시키는 열석발언권을 행사해 중 | 한 있다. 이 있다. 이 대 전 이 있다. 이 있다. 이 | 0.0000	False
    --------------------------------------------------------------------------------
    [2000/100000] Train loss: 3.91365, Valid loss: 5.63714, Elapsed_time: 785.56412
    Current_accuracy : 0.100, Current_norm_ED  : 0.19
    Best_accuracy    : 0.100, Best_norm_ED     : 0.23
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 다. 이 한 대해 10000 10일 10일 이 전 전 전 전 전 전 전 전 전 | 0.0000	False
    선고다. 글랜드 체스터 지방법원이 페이스북에 폭동을 선동하는 | 로 있다. 이 대 전 전 전 전 전 전 전 전 전 전 전 전 있다. 이 | 0.0000	False
    성은 배제할 수 없을것으로 보여 향후 검찰의 수사가 주목된다. | 다. 이 전 전 전 전 있다. 이 했다. 것 했다. 것 했다. 것 | 0.0000	False
    해 노사정 사회협약 시스템을 만들어 갈 게획을 분명히 가지고 있다" | 다. 이 대 전 전 전 전 전 전 전 전 전 있다. 이 있다. 이 했다. 이 있다 | 0.0000	False
    유동업조합으로 시작한 서울우유는 이후 국내 낙농산업의 신을 선 | 서 있다. 이 대은 있다. 이 "고 있다. 이 "고 있다. 이 대 전 전 전 전  | 0.0000	False
    --------------------------------------------------------------------------------
    [2500/100000] Train loss: 3.66017, Valid loss: 5.60662, Elapsed_time: 978.89272
    Current_accuracy : 0.100, Current_norm_ED  : 0.22
    Best_accuracy    : 0.100, Best_norm_ED     : 0.23
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    제 상황처럼 해볼 수 있게 돼정비 전문인력 양성은 물론, 새로운 정비 | 로 있다. 이 한 한 한 한 한 한 있는 있다. 말했다. 또 대한 있 | 0.0000	False
    화축제가 전주시 풍남동 경기전 일대에서 열린다. 임실군에선 30일 | 대한 300 일 대표 지 한 대 한 대원에 대한 20000년 | 0.0000	False
    해 노사정 사회협약 시스템을 만들어 갈 게획을 분명히 가지고 있다" | 로 대해 사으로 대해 주 있다. 말했다. 말했다. 말했다. 말 | 0.0000	False
    끝에 확정습니다.감독 경질까지 겪어야 했던 대표팀의 상처 많은 | 한  사원에 대한 "고 있다. 이 한 대원은 대해 주 한 대원 | 0.0000	False
    심야까지 회의를 열지 못하고 24일로 연기했다. 법안심사소위 위원 | 정정이에서 대한 의 한 대원 의 한 대원 지 한 대원 이 한 대 | 0.0000	False
    --------------------------------------------------------------------------------
    [3000/100000] Train loss: 3.33941, Valid loss: 5.71458, Elapsed_time: 1171.64891
    Current_accuracy : 0.100, Current_norm_ED  : 0.22
    Best_accuracy    : 0.100, Best_norm_ED     : 0.23
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    사 등 다른 사안과 연계 가능성까지 언급해 민주당의 반발을 고, 북 | 다. 것은 다. 검찰은 대해 전 전 전 전 전 전 전 전 전 전 전 전 | 0.0000	False
    . 내일은 오늘보다 기온이 더 높습니다. 서울 아침 기온 영하 6도, | 의 있다. 지 했다. 문 전 전 전 전 전 전 전 전 있다. | 0.0000	False
    물의를 빚어 감찰 대상에 올던 c검사가 이날 사표를 제출했다고 밝 | 조하는 것이 기사가 이 전 사장에 대한 조사가 이 한 한 가 가 가 가 | 0.0000	False
    기관은 "대내외 경제여건과 금시장의 불확실성이 지속되는 만큼 정 | 서는 경 전사사사 이 전 전 전 전 지사를 지해 다는 사는 사 | 0.0000	False
    다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 다. 것이 이다. 이다. 이 가 가 전 전 전 전 전 전 전 전 전 전 10 | 0.0000	False
    --------------------------------------------------------------------------------
    [3500/100000] Train loss: 2.94029, Valid loss: 5.62233, Elapsed_time: 1365.77678
    Current_accuracy : 0.200, Current_norm_ED  : 0.24
    Best_accuracy    : 0.200, Best_norm_ED     : 0.24
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 한이 강해 예에 대한 사가 지지가 이다면 이씨에 이라면 지씨가 지 | 0.0000	False
    9월 중국전 한 골 차 진땀승을 시작으로, 조 최약체로 꼽히던 시리아 | 서는 중 전 전 전 전 없 대표는 대표 지시 지시 지지, 가 있다. 한 한 전 | 0.0000	False
    대구지법 제 12형사부(재판장 김종필 부장판사)는 1일 생활고로 남 | 있다. 20사 대원이 이라 사재가 가 한다. 검찰은 사고 사 | 0.0000	False
    데 인식을 같이했다"고 밝다. 양 기관은 또 "거시정책은 물가안정에 | 한인인을 받다"고 말했다. 또 "고 있다. 또 "정부의 안안 소정정부에 | 0.0000	False
    다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 다. 일 기관관자에 대해 10000원 의 처리 기리를 하다. 검찰은 다. | 0.0000	False
    --------------------------------------------------------------------------------
    [4000/100000] Train loss: 2.49062, Valid loss: 5.57703, Elapsed_time: 1562.52009
    Current_accuracy : 0.300, Current_norm_ED  : 0.27
    Best_accuracy    : 0.300, Best_norm_ED     : 0.27
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    끼치고 있다. 하지만 수자원공사는 공기업 경영평가에서는 작성기준 | 고치하 고 있다. 한다는 검찰은 사실하는 "일일 수안을 강기하는 수 | 0.0000	False
    비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 법이 일본 조사에 대표를 받다. 또 대표는 통동조 구통을 통해해 해정해 | 0.0000	False
    귀국치 않다.                   | 로 수 있다.                   | 0.0049	False
    족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | . 전체가서는 시일 일도도 등도 등 전 있으로 서울 그동 등 전 전 | 0.0000	False
    념관 강당에서 발대식을 갖고, 4일부터 13일까지 9박 10일간의 여정 | 원한 '위해 의 의원을 열한 것으로 선해했다. 이 대표은 법의 동안 | 0.0000	False
    --------------------------------------------------------------------------------
    [4500/100000] Train loss: 2.02598, Valid loss: 5.35943, Elapsed_time: 1756.87213
    Current_accuracy : 0.200, Current_norm_ED  : 0.32
    Best_accuracy    : 0.300, Best_norm_ED     : 0.32
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    내일 아침 기온은 서울 12도, 광주 14도로 예상니다. 낮 기온은 서 | 의도 아이 기온은 서울 1도도 주주도, 한 기온도 서울 12도 오는 1 | 0.0000	False
    파이시한 풍미와 미네의 느낌이 아주 조화롭게 어우러져 있는 와인 | 지사기에 대해 100일 은인지 생생을 받은 것으로 있어 것으로 있는 | 0.0000	False
    파도가 높고 날이 어두워지자 이날 구조작업은 철수하고 17일 오전 | 의 일은 등 있지가 한한 것으로 강어하고 있다. 지난 1년년 일 | 0.0000	False
    이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 의 서제가 남우우 일 은행의 모울 사도에 다러한 것으로 보 | 0.0000	False
    망각하는 것이며 국민적 평화통일 의지를 손하는 과오를 범하는 것 | 정하는 것이라며 것이 인화하는 의를 하는 하는 하는 하는 것 | 0.0000	False
    --------------------------------------------------------------------------------
    [5000/100000] Train loss: 1.60007, Valid loss: 5.81849, Elapsed_time: 1950.13207
    Current_accuracy : 0.100, Current_norm_ED  : 0.30
    Best_accuracy    : 0.300, Best_norm_ED     : 0.32
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 사생이 지사지 대해 10월 만 대표 동안지 사반한 기반한 기동하 | 0.0000	False
    건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 건가 강화했다. 이씨에 현현 100의 의 이은 날 강도은 이날 도도 | 0.0000	False
    어우러져 있는 그랑 마레농은 장어의 느끼한 맛을 잘 잡아준다. 또한 | 수수계라는 그은 보기수 이제는 한 생권이 기고 있다"고 말했다. 한 | 0.0000	False
    이는 방식을 택해 공장식 축산을 할 수 없다며 한정생산을 강점으로 | 이는 것이라 한 "사회에 수사 이 없기 전 생생 5월 점기를 점 | 0.0000	False
    흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 조리는 행위"라고 이어 북한 법안는 "이라고 한내 원은  장시 법 | 0.0000	False
    --------------------------------------------------------------------------------
    [5500/100000] Train loss: 1.22382, Valid loss: 4.81803, Elapsed_time: 2143.86801
    Current_accuracy : 0.700, Current_norm_ED  : 0.43
    Best_accuracy    : 0.700, Best_norm_ED     : 0.43
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    있다"고 말했다. 또한 안 팀장은 "정부는 일본 정부에 일본군 위안부 | 있다"고 말했다. 또 안 전 평정부터 정부는 일본군 위해 정부를 일본군 | 0.0000	False
    정이다. 은행권역에서는 은행감독국이 신설되고 외환업무실이 외환 | 정이다. 대한 법안을 받은 것으로 알주된 등에서 신소되 | 0.0000	False
    계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 은는 이 전 있다며 지난 이도 오도 마양지 마천도가 이어도 이씨에 | 0.0000	False
    을 감안해 집행유예를 선고한다"고 밝다. 노씨는 2003년 10월께 | 을 감안해 집행 2011년 10고 의실했다. 20월씨가 신월 20월께 | 0.0000	False
    한 일본대사관 앞의 일본군 위안부 평화비(소녀상)에 '다시마(독도 | 한 일본대사회의 의 일본군 개위을 열영지 10일)에 대해 시합으로 | 0.0000	False
    --------------------------------------------------------------------------------
    [6000/100000] Train loss: 0.90711, Valid loss: 4.35496, Elapsed_time: 2338.87401
    Current_accuracy : 3.200, Current_norm_ED  : 0.51
    Best_accuracy    : 3.200, Best_norm_ED     : 0.51
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 표로 광해방지 기술 중 오염토양 원원기술, 배산배수 처리기술 수표를 자 | 0.0000	False
    입장을 발표하고, 사회적으로 손가락질받을 일이 학교에서 발생해 안 | 입장을 발해하고, 사회적으로 사가를 받지 조안에 안한 1안원인 발생 | 0.0000	False
    아들 c군은 주소지의 학교를 피하기 위해 서울 강동구로 위장전입한 | 아들  동은 총위의 피이 학입니이 앞서 재주한 강위 전입가 경정 | 0.0000	False
    육관광부 우수축제로 선정된 '제81회 춘향제'가 열린다. 정시에서 | 감관관부 우수축제로 선정된 '제선선회 가향제'제 열린다. 정시에서 | 0.0000	False
    두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 두씨씨는 "행사 수익을 전액해 유니세면에 연부하 부 했다. 사진집 | 0.0000	False
    --------------------------------------------------------------------------------
    [6500/100000] Train loss: 0.64602, Valid loss: 3.24961, Elapsed_time: 2534.10306
    Current_accuracy : 14.100, Current_norm_ED  : 0.66
    Best_accuracy    : 14.100, Best_norm_ED     : 0.66
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    민생인권법에 대해 "민주당이 북한 주민의 민생문제를 제기한 바 있 | 민생인권법에 대해 "민주당이 북한 주민의 민민문제를 제기한 것 있 | 0.0000	False
    니다. 날이 풀리면서 내일 오전에 미세먼지 농도가 높아질 수 있어 주 | 니다. 날이 보리면서 내일 오전에 미세지지 농도가 높아 수 있어 있 | 0.0000	False
    대피해 있다.                   | 대해 않다.                    | 0.0055	False
    에 오른다. 이 기간 동안 방단은 블라디보스 한인 집단 거주지, 안 | 과 오어 오 주간에 대한 방들을 들단 사로에 한다. 하기 보일 예 | 0.0000	False
    뒤 취한 조니다. 법원에 청구한 동결 대상 재산은 28억 원에 매입한 | 한 취한 조니다. 법원에 청구한 동결대한 경제한 결원은 최니한 매에 | 0.0000	False
    --------------------------------------------------------------------------------
    [7000/100000] Train loss: 0.43400, Valid loss: 3.30518, Elapsed_time: 2729.21045
    Current_accuracy : 17.400, Current_norm_ED  : 0.66
    Best_accuracy    : 17.400, Best_norm_ED     : 0.66
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    소재로 한 '2011 고창 복분자 푸드페스티벌'을 개최해 축제의 계절에 | 정씨를 시고 두두의 체제를 통통해 성제와 지사에서 '최기 전 의에 대기업 | 0.0000	False
    . 타이어는 재질 역시 스노우 타이어의 제동력을 높이는 역할을 하게 | . 타이어는 재재 역시 스노우 타이어의 제동력을 높이는 역할을 하게 | 0.0000	False
    검찰이 박근혜 전 대통령에 대한 재산 동결 조치를 취했습니다. 내곡 | 검찰이 5근혜 전 전 사회가 지상하 습니다. 이르는 동결 공고가 했다. 2년 | 0.0000	False
    우유, 행복한 고객'을 실현하다고 선언했다. 75년을 넘어 100년으 | 우유, 행복한 고객'을 실현하다고 선했했다. 75년을 넘어 100년으 | 0.0000	False
    계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 이는 박이 기온와 목월 4도 기태자 몰을 징탁하고, 것으로 안 해 | 0.0000	False
    --------------------------------------------------------------------------------
    [7500/100000] Train loss: 0.28885, Valid loss: 1.92419, Elapsed_time: 2922.57164
    Current_accuracy : 49.600, Current_norm_ED  : 0.82
    Best_accuracy    : 49.600, Best_norm_ED     : 0.82
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    다. 조 전 수석은 박 전 대통령이 이 부회장 퇴진을 지시했고 손경식 c | 다. 조 전 수석은 박 대유 조 위원에 대부를 이진했다. 이시 | 0.0000	False
    에서 개봉한다. 일본 산이스포츠는 21일 "'마이 브라더('우리형'의 | 에서 강한한다. 일본 오후스리 '우우일 '일 마관이라 '화 모형'의 | 0.0000	False
    성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 성재가 환자 의료 및로 지원 및 불약 여성, 취약계 지료 비가 사가 사업 | 0.0000	False
    인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 0.0060	True
    받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 0.0477	True
    --------------------------------------------------------------------------------
    [8000/100000] Train loss: 0.20194, Valid loss: 1.61713, Elapsed_time: 3115.54133
    Current_accuracy : 64.300, Current_norm_ED  : 0.85
    Best_accuracy    : 64.300, Best_norm_ED     : 0.85
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    을 감안해 집행유예를 선고한다"고 밝다. 노씨는 2003년 10월께 | 을 채용해 집행을 집선한 노계적으로 선고했다. 노정과정. 10월 의 | 0.0000	False
    감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 0.0954	True
    원은 옛 삼성동 자택을 67억 원에 팔면서 생긴 돈으로 파악하고 있습 | 원은 옛 삼성동 자택을 67억 원에 팔면서 생긴 돈으로 파악하고 있습 | 0.0001	True
    단과 총격을 벌이다 경찰이 사망하기도 했다.  | 단과 총격을 벌이다 경찰이 사망하기도 했다.  | 0.1776	True
    사 사장은 연임이 유력한 것으로 알려다. 공기업 경영평가에서 a등 | 사 사장은 연연이 유려한 것을 선구했다. 이기 정은 20일 간대 | 0.0000	False
    --------------------------------------------------------------------------------
    [8500/100000] Train loss: 0.13990, Valid loss: 3.38108, Elapsed_time: 3307.78651
    Current_accuracy : 26.700, Current_norm_ED  : 0.70
    Best_accuracy    : 64.300, Best_norm_ED     : 0.85
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 에서 "기자관을 놓 것으로 부장해 1박기, 수미순을 논어할 것을 사음를 | 0.0000	False
    은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 은 "세계 때위의 수구 수4으로부 전 전 시4이라로 선려했다. 최근 | 0.0000	False
    당하는 시공사 관계자도 붕괴위험을 사전에 지적하며 보강공사를 수 | 당하는 시공사 관계자도 붕괴위다. 사적에 따해 재동보관 사을 사 | 0.0000	False
    한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 0.0001	True
    면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 면 그 압력 때문에 수분이 발생하게 되는데, 바퀴가 닫는 면적이 넓어 | 0.0429	True
    --------------------------------------------------------------------------------
    [9000/100000] Train loss: 0.10546, Valid loss: 1.26386, Elapsed_time: 3500.25822
    Current_accuracy : 74.600, Current_norm_ED  : 0.89
    Best_accuracy    : 74.600, Best_norm_ED     : 0.89
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    식이 열리는 저녁에는 체감온도가 영하 10도 안팎으로 떨어질 것으 | 식이 열리는 저녁에는 체감도도안에는 것으로 도에 다. 또리를 찾 | 0.0000	False
    는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 0.0391	True
    니다. 현재 중부 지방과 영남을 중심으로 건조 특보가 확대되고 있습 | 니다  현재 중부 지방과 영남을 중심으로 건조 특보가 확대되고 있습 | 0.0000	False
    에 입맛이 떨어지기 십상인 여름에 더할 나위 없이 좋다. 초복을 시작 | 에 입맛이 떨어지기 십상인 여름에 더할 나위 없이 좋다. 초복을 시작 | 0.0327	True
    자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 0.0113	True
    --------------------------------------------------------------------------------
    [9500/100000] Train loss: 0.08152, Valid loss: 1.30192, Elapsed_time: 3693.88730
    Current_accuracy : 74.900, Current_norm_ED  : 0.89
    Best_accuracy    : 74.900, Best_norm_ED     : 0.89
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    어지기를 바란다"고 말했다. 한편, 신한카드는 이날 차병원에 소아뇌 | 가가 추동했다. 바난 "고운당 이날다. 또편 "아당 10에 대한 스위에서 | 0.0000	False
    개국 8개 법인에 파견하기로 했다. 이외에도 대한통운은 다양한 사내 | 개국 8개 법안에 강견하기로 했다. 이외에도 대통통을 보내다면 정내이 있 | 0.0000	False
    부채 문제 등 취약요인에 대한 적극적인 대응이 필요한 시점이라는 | 부채 문제 등 취약요인에 대한 적극적인 대응이 필요한 시점이라는 | 0.3055	True
    진군행렬 등 다양한 볼거리와 즐길거리를 제공할 계획이다. 고창군 | 진관행를 갖질한 것요리리를 갖거한 계위를 강리한 계획이다. 고리군 | 0.0000	False
    받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 0.3332	True
    --------------------------------------------------------------------------------
    [10000/100000] Train loss: 0.06092, Valid loss: 1.30885, Elapsed_time: 3887.31226
    Current_accuracy : 75.300, Current_norm_ED  : 0.89
    Best_accuracy    : 75.300, Best_norm_ED     : 0.89
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 0.2752	True
    교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 0.2238	True
    있었을 것이라면서 김 전 회장의 입국설에 회의적인반응을 보다. | 있었을 것이라면서 김 전 회장의 입국설에 회의적인반응을 보다. | 0.3019	True
    상이한 광해방지 제도 및 적용기술에 대해 한국녹색산업진흥협회와 | 상이한 광해방지 제도 및 적용기술에 대해 한국녹색산업진흥협회와 | 0.0418	True
    많습니다. 남해안은 새벽부터 아침 사이에 눈 또는 비가 내리습 | 많습니다. 남해안은 새벽부터 아침 사이에 눈 또는 비가 내리습 | 0.2367	True
    --------------------------------------------------------------------------------
    [10500/100000] Train loss: 0.05519, Valid loss: 1.28177, Elapsed_time: 4080.30809
    Current_accuracy : 76.300, Current_norm_ED  : 0.89
    Best_accuracy    : 76.300, Best_norm_ED     : 0.89
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    와의 동일여부 등을 확인할 예정이다.      | 와의 동일여부 등을 확인할 예정이다.      | 0.5552	True
    어학교육 프로그램을 운영하고 있다. 희망자들을 대상으로 토익, 중 | 있어어이 크로 보랑은 과제하기 보도, 보영하는 것으이 보고, 과 과과 | 0.0000	False
    의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 의 500만원'이라고 일치 공주장을 하는 것과 관련해 진술 은폐를 위 | 0.0047	False
    과 다르게 경영실적보고서를 작성해 높은 등급을 받다. 이 덕분에 | 과 다르게 경영실적보고서를 작성해 높은 등급을 받다. 이 덕분에 | 0.4516	True
    비정규직법안에 대한 국회 처리를 둘러싸고 여당과 노동계가 한 | 비정규직법안에 대한 국회 처리를 둘러싸고 여당과 노동계가 한 | 0.3361	True
    --------------------------------------------------------------------------------
    [11000/100000] Train loss: 0.03855, Valid loss: 1.30421, Elapsed_time: 4279.02196
    Current_accuracy : 73.800, Current_norm_ED  : 0.89
    Best_accuracy    : 76.300, Best_norm_ED     : 0.89
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    성가족위원회 소속 이자스민 의원은 '일본군 위안부 피해자들에게 법 | 성가족위원회 소속 이자스민 의원은 '일본군 위안부 피해자들에게 법 | 0.2506	True
    년 가까이 끌어온 북한인권법을 이번 임시국회에서 반드시 처리하 | 년 가까이 끌어온 북한인권법을 이번 임시국회에서 반드시 처리하 | 0.3663	True
    지하철 7호선 방화사건을 수사중인 경기도 광명경찰서는 사건당일 | 지하철 7호선 방화사건을 수사중인 경기도 광명경찰서는 사건당일 | 0.0698	True
    j그룹 회장에게 그 뜻을 전달했다고 진술했습니다. kbs 뉴스 이현기 | j그룹 회장에게 그 뜻을 전달했다고 진술했습니다. kbs 뉴스 이현기 | 0.4731	True
    끼치고 있다. 하지만 수자원공사는 공기업 경영평가에서는 작성기준 | 끼치고 있다. 하지만 수자원공사는 공기업 경영평가에서는 작성기준 | 0.2898	True
    --------------------------------------------------------------------------------
    [11500/100000] Train loss: 0.05134, Valid loss: 1.18258, Elapsed_time: 4475.62366
    Current_accuracy : 78.300, Current_norm_ED  : 0.90
    Best_accuracy    : 78.300, Best_norm_ED     : 0.90
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    경영실적보고서를 평가편람 내용과 다르게 작성하지 말라"고 주의를 | 경영실적보고서를 평가편람 내용과 다르게 작성하지 말라"고 주의를 | 0.4152	True
    은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 0.1273	True
    로 정하고 추진했다. 이들 사업에 도는 사업당 200억 원을 한도로 20 | 로 정하고 추진했다. 이들 사업에 도는 사업당 200억 원을 한도로 20 | 0.6166	True
    대한 재검토와 향후대책을 논의하는 등 비상 대기상태에 들어다. | 대한 재검토와 향후대책을 논의하는 등 비상 대기상태에 들어다. | 0.0137	True
    비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 비와 낙동강 사업에 따른 오염 토양환 안안정환 환사에 대해 | 0.0002	False
    --------------------------------------------------------------------------------
    [12000/100000] Train loss: 0.04623, Valid loss: 1.18507, Elapsed_time: 4670.81722
    Current_accuracy : 78.300, Current_norm_ED  : 0.90
    Best_accuracy    : 78.300, Best_norm_ED     : 0.90
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    검찰은 글로스가 몽골 금광 개발에 투자한다는 등의 허위 공시를 | 검찰은 글로스가 몽골 금광 개발에 투자한다는 등의 허위 공시를 | 0.3373	True
    이 곧 글로벌 기업이  수 있다는 의지가 담겨있다고 회사측은 설명 | 이 곧 글로벌 기업이  수 있다는 의지가 담겨있다고 회사측은 설명 | 0.3455	True
    으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 0.2591	True
    개국 8개 법인에 파견하기로 했다. 이외에도 대한통운은 다양한 사내 | 개국 8개 법안에 사견한 상인하다. 앞원에로 대한대면 보내대에서 상인 | 0.0000	False
    급할 가능성이 높다. 3억원을 개인적으로 사용하지 않다면 모든  | 급할 가능성이 높다. 3억원을 개인적으로 사용하지 않다면 모든  | 0.2749	True
    --------------------------------------------------------------------------------
    [12500/100000] Train loss: 0.05124, Valid loss: 1.22571, Elapsed_time: 4866.13134
    Current_accuracy : 77.700, Current_norm_ED  : 0.90
    Best_accuracy    : 78.300, Best_norm_ED     : 0.90
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    맞는 이 축제는 공음면 학원농장의 보리밭에서 5월 8일까지 열린다. | 맞는 이 축제는 공음면 학원농장의 보리밭에서 5월 8일까지 열린다. | 0.4528	True
    김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 0.1637	True
    다 습지조성 용역예산 삭감 역시 김 지사가 남강댐물 부산공급의 | 다. 데 조사가 지난 7산기때터 지시시 공사가 보물조사에 | 0.0000	False
    으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 0.2010	True
    다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 0.0718	True
    --------------------------------------------------------------------------------
    [13000/100000] Train loss: 0.02351, Valid loss: 1.19419, Elapsed_time: 5060.03268
    Current_accuracy : 78.600, Current_norm_ED  : 0.91
    Best_accuracy    : 78.600, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    문제 해결과 한반도 평화를 일구도록 노력하다고 약속했습니다. 지 | 문제 해결과 한반도 평화를 일구도록 노력하다고 약속했습니다. 지 | 0.7088	True
    민생인권법을 북한인권법과 병합처리하자고 주장하는 것은 본질을 | 민생인권법을 북한인권 과 처합처처 처합처합한 의의 보질하는 북은 | 0.0000	False
    감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 0.0440	True
    내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 0.1846	True
    건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 0.5028	True
    --------------------------------------------------------------------------------
    [13500/100000] Train loss: 0.04880, Valid loss: 1.14842, Elapsed_time: 5253.52071
    Current_accuracy : 79.300, Current_norm_ED  : 0.91
    Best_accuracy    : 79.300, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    일부 보도에 대해 "사실이 아니다"라고 밝다. 김형곤은 14일 오후 | 일부 보도에 대해 "사실이 아니다"라고 밝다. 김형곤은 14일 오후 | 0.6292	True
    많습니다. 남해안은 새벽부터 아침 사이에 눈 또는 비가 내리습 | 많습니다. 남해안은 새벽부터 아침 사이에 눈 또는 비가 내리습 | 0.1714	True
    나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 0.4060	True
    주민의 민생 문제도 그대로 반영된다"고 덧붙다. 이처럼 여야가 북 | 주민의 민생 문제도 그대로 반영된다"고 덧붙다. 이처럼 여야가 북 | 0.1689	True
    국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 0.4528	True
    --------------------------------------------------------------------------------
    [14000/100000] Train loss: 0.03131, Valid loss: 1.08527, Elapsed_time: 5448.52078
    Current_accuracy : 80.400, Current_norm_ED  : 0.91
    Best_accuracy    : 80.400, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    올린 것에 대해서는 지나친 중형"이라고 주장했다. 반면 공공의 안녕 | 올며 것에 대해게 조사장을 주진했 것도 후주장는 반고 있고기 | 0.0000	False
    창 올림픽은 88올림픽 이후 성장과 발전을 전세계에 알리는 좋은 계 | 창 올림픽은 88올림픽 이후 성장과 발전을 전세계에 알리는 좋은 계 | 0.7149	True
    추가로 확보해 조사중이다. 경찰에 따르면 5일 밤 9시51분께 강원도 | 추가로 확보해 조사중이다. 경찰에 따르면 5일 밤 9시51분께 강원도 | 0.7648	True
    종 선정 등 이 두가지로 한정돼 있다"며 "(내가 경제수석시절에)동반 | 종 선정 등 이 두가지로 한정돼 있다"며 "(내가 경제수석시절에)동반 | 0.3445	True
    이 죽어다"며 "문 후보는 당시 민정수석으로 공권력 집행의 최종 결 | 이 죽어다"며 "문 후보는 당시 민정수석으로 공권력 집행의 최종 결 | 0.0046	True
    --------------------------------------------------------------------------------
    [14500/100000] Train loss: 0.01230, Valid loss: 1.18131, Elapsed_time: 5644.43900
    Current_accuracy : 80.100, Current_norm_ED  : 0.91
    Best_accuracy    : 80.400, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 0.5307	True
    으로 모실 예정"이라고 말했다. 문 후보는 또 "참여정부는 한계가 있 | 으로 모실 예정"이라고 말했다. 문 후보는 또 "참여정부는 한계가 있 | 0.6810	True
    검찰 관계자가 전했다. c검사는 "아들이 해외유학 때문에 고교 진학 | 검찰 관계자가 전했다. c검사는 "아들이 해외유학 때문에 고교 진학 | 0.4709	True
    비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 0.2535	True
    호사는 박 전 대통령의 현금 10억여 원도 보관 중입니다. 지금까지 드 | 호사는 박 전 대통령의 현금 10억여 원도 보관 중입니다. 지금까지 드 | 0.5006	True
    --------------------------------------------------------------------------------
    [15000/100000] Train loss: 0.01825, Valid loss: 1.19638, Elapsed_time: 5839.22081
    Current_accuracy : 80.500, Current_norm_ED  : 0.91
    Best_accuracy    : 80.500, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    효중인 가운데 물결이 거세게 일습니다. 내일 아침에는 서울의 기 | 효중인 가운데 물결이 거세게 일습니다. 내일 아침에는 서울의 기 | 0.8153	True
    영장을 청구했으며 긴급체포돼 수감 중인 은 전 위원을 이날 다시 불 | 영장을 청구했으며 긴급체포돼 수감 중인 은 전 위원을 이날 다시 불 | 0.1324	True
    전북도를 찾는 관광객에게 오랫동안 기억에 남을 추억거리를 선물할 | 전북도를 찾는 관광객에게 오랫동안 기억에 남부 추억를 거거를 | 0.0031	False
    념관 강당에서 발대식을 갖고, 4일부터 13일까지 9박 10일간의 여정 | 념관 강당에서 발대식을 갖고, 4일부터 13일까지 9박 10일간 | 0.0006	False
    계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 0.3749	True
    --------------------------------------------------------------------------------
    [15500/100000] Train loss: 0.05724, Valid loss: 1.11099, Elapsed_time: 6034.78470
    Current_accuracy : 80.500, Current_norm_ED  : 0.91
    Best_accuracy    : 80.500, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 0.6809	True
    감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 0.1393	True
    인 피고인이 잘못을 뉘우치며 정신적 후유증에 시달리고 있는 점 등 | 인 피고인성 잘못을 뉘치하고 밝다. 박씨가 증리 지시적을 잘 할 | 0.0000	False
    장을 고발했지만 무의 처리다. 당시 오 교사는 아는 검사를 통해 | 장을 고발했지만 무의 처리다. 당시 오 교사는 아는 검사를 통해 | 0.7238	True
    아들 c군은 주소지의 학교를 피하기 위해 서울 강동구로 위장전입한 | 아들 c군은 주소지의 학교를 피하기 위해 서울 강동구로 위장전입한 | 0.6722	True
    --------------------------------------------------------------------------------
    [16000/100000] Train loss: 0.01058, Valid loss: 1.12407, Elapsed_time: 6229.34202
    Current_accuracy : 80.600, Current_norm_ED  : 0.91
    Best_accuracy    : 80.600, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    운영할 필요가 있다는 데 공감했다"며 "특히 수출과 내수간 격차, 지 | 운영할 필요가 공지하 대표를 하는 중 수사 등격에서 할차, 시가지, 지 | 0.0000	False
    진군행렬 등 다양한 볼거리와 즐길거리를 제공할 계획이다. 고창군 | 진행행를 다양한 계능을 강요한 것을 안수하기 자위를 추련할 계 | 0.0000	False
    채용할 경우 증원인력 1인당 월 50만원(올해 60만원)씩한시적으로 | 채용할 경우 증원인력 1인당 월 50만원(올해 60만원)씩시시적으로 | 0.1736	False
    원은 옛 삼성동 자택을 67억 원에 팔면서 생긴 돈으로 파악하고 있습 | 원은 옛 삼성동 자택을 67억 원에 팔면서 생긴 돈으로 파악하고 있습 | 0.5507	True
    에 소재파악 수사를 의해 상태"라며 "만약 김 전 회장이 국내에 | 에 수번파 수수의 소태를 벌러 만약 "정를 밝다. 김 200 내내 | 0.0000	False
    --------------------------------------------------------------------------------
    [16500/100000] Train loss: 0.00967, Valid loss: 1.09822, Elapsed_time: 6424.44344
    Current_accuracy : 80.200, Current_norm_ED  : 0.91
    Best_accuracy    : 80.600, Best_norm_ED     : 0.91
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 0.7407	True
    했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 0.4390	True
    외은지점감독실이 신설된다. 중소서민금 권역에서는 여신전문서 | 외은지점감독실이 신설된다. 중소서민금 권역에서는 여신전문서 | 0.6962	True
    러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 0.5853	True
    내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 0.3948	True
    --------------------------------------------------------------------------------
    [17000/100000] Train loss: 0.10773, Valid loss: 1.10656, Elapsed_time: 6617.53951
    Current_accuracy : 82.000, Current_norm_ED  : 0.92
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 0.7160	True
    제 상황처럼 해볼 수 있게 돼정비 전문인력 양성은 물론, 새로운 정비 | 제 상황처럼 해볼 수 있게 돼정비 전문인력 양성은 물론, 새로운 정비 | 0.0917	True
    편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 0.4789	True
    장을 고발했지만 무의 처리다. 당시 오 교사는 아는 검사를 통해 | 장을 고발했지만 판상을 통는 조 평에 주류, 당의 올다. 지는 올 | 0.0000	False
    럴 플랜'를 출시했다. 옥수수를 주원료로 한 배합사료 대신 풀을 먹 | 럴을 구립된다. 최자 적은 예시 투무와 사건 복위를 먹료 등은 사시를 | 0.0000	False
    --------------------------------------------------------------------------------
    [17500/100000] Train loss: 0.00883, Valid loss: 1.14472, Elapsed_time: 6810.34346
    Current_accuracy : 81.100, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    도록 모든 노력을 아끼지 않을 것"이라고 주장했다. 이에 앞서 지난 1 | 도록 모든 노력을 아끼지 않을 것"이라고 주장했다. 이에 앞서 지난 1 | 0.7871	True
    이 각각 생보검사국와 손보검사국으로 바뀌고 보험감독국이 분리된 | 이 각각 생보검사국와 손보검사국으로 바뀌고 보험감독국이 분리된 | 0.6142	True
    럴 플랜'를 출시했다. 옥수수를 주원료로 한 배합사료 대신 풀을 먹 | 럴 플을 를 "고할 적으로 사 없사가 보험복이 없태로를 보고 한 때 | 0.0000	False
    오리엔테이션 자리에서 16학번 한 남학생이 신입생 후배에게 16학번 | 오리엔테이션 자리에서 16학번 한 남학생이 신입생 후배에게 16학번 | 0.5794	True
    각 중단하고 박 원내대표에게 의가 있다면 당당히 기소하라"며 "한 | 각 중단하고 박 원내대표에게 의가 있다면 당당히 기소하라"며 "한 | 0.3818	True
    --------------------------------------------------------------------------------
    [18000/100000] Train loss: 0.00813, Valid loss: 1.14786, Elapsed_time: 7001.29847
    Current_accuracy : 80.800, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 0.5245	True
    밝다. 박씨도 로비 자체를 부인하고 있다. 박씨는 부산저축은행에 | 밝다. 박씨도 로비 자체를 부인하고 있다. 박씨는 부산저축은행에 | 0.8818	True
    대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 0.7837	True
    보했지만 c검사는 "말미를 달라"며 출석을 미뤄온 것으로 알려다. | 보했지만 c검사는 "말미를 달라"며 출석을 미뤄온 것으로 알려다. | 0.0465	True
    인 16일 휴대전화 위치 추적을 벌인 결과 현 전 의원과 동선이 일치하 | 인 16일 휴대전화 위치 추적을 벌인 결과 현 전 의원과 동선이 일치하 | 0.6188	True
    --------------------------------------------------------------------------------
    [18500/100000] Train loss: 0.00674, Valid loss: 1.28581, Elapsed_time: 7193.92155
    Current_accuracy : 77.400, Current_norm_ED  : 0.90
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    심야까지 회의를 열지 못하고 24일로 연기했다. 법안심사소위 위원 | 심야까지 회의를 열지 못하고 24일로 연기했다. 법안심사소위 위원 | 0.2222	True
    감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 0.3014	True
    입니다.                      | 했다.                       | 0.2687	False
    한국노총도 법안 강행시 벌이기로 했던 노사정위 탈퇴와 대정부 투쟁 | 한국노총도 법안 강역시 벌이 1도 있도 보사위원 탈퇴 정책도 거치 | 0.0000	False
    제하는 것은 표현의 자유를 가로막을 뿐이며 정부가 반정부 시위 때 | 제하는 것은 표현의 자유를 가로막을 뿐이며 정부가 반정부 시위 때 | 0.4337	True
    --------------------------------------------------------------------------------
    [19000/100000] Train loss: 0.00700, Valid loss: 1.20055, Elapsed_time: 7387.08892
    Current_accuracy : 80.100, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    만 아니라 전후좌우 방향으로 미끄러지는 것을 방지하는 효과가 있다 | 만 아니라 전후좌우 방향으로 미끄러지는 것을 방지하는 효과가 있다 | 0.2630	True
    외은지점감독실이 신설된다. 중소서민금 권역에서는 여신전문서 | 외은지점감독실이 신설된다. 중소서민금 권역에서는 여신전문서 | 0.7878	True
    연두색으로 표시된 많은 지역도 강풍 주의보가 발효중인데요. 실제로 | 연두색으로 표시된 많은 지역도 강풍 주의보가 발효중인데요. 실제로 | 0.7470	True
    체 호워드 리그의 드루 닐슨은 "징역 4년형은 흉기로 상해를 입 | 체 호워드 리그의 드루 닐슨은 "징역 4년형은 흉기로 상해를 입 | 0.5793	True
    마당 등을 선보일 계획이다. 남원에선 5월 6일부터 10일까지 문화체 | 마당 등을 선보하기고 이다. 5원에선 선부전 전씨에 지정한 | 0.0001	False
    --------------------------------------------------------------------------------
    [19500/100000] Train loss: 0.00929, Valid loss: 1.21807, Elapsed_time: 7577.78864
    Current_accuracy : 80.300, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 0.4126	True
    원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 0.7965	True
    인터넷을 차단했던 이집트 정부처럼 행동하려 한다고 비난했다.  | 인터넷을 차단했던 이집트 정부처럼 행동하려 한다고 비난했다.  | 0.6238	True
    니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 0.8854	True
    감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 0.8942	True
    --------------------------------------------------------------------------------
    [20000/100000] Train loss: 0.03443, Valid loss: 1.22863, Elapsed_time: 7771.24379
    Current_accuracy : 78.600, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 일해 보양식을 | 0.0002	False
    사용할 계획이다. 이재우 사장은 "이번 협약 체결로 병마와 싸우고 | 사용할 계획이다. 이재우 사장은 "이번 협약 체결로 병마와 싸우고 | 0.6623	True
    는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 는 5월 7일부터 8일까지 동학농민 명의 의의를 되새기는 '제44회 | 0.4981	True
    로 수사를 확대하고 있다. 부산지검 공안부(부장검사 이태승)는 3억 | 로 수사를 확대하고 있다. 부산지검 공안부(부장검사 이태승)는 3억 | 0.5774	True
    등도 수행하게 되며, 이 과정을 마치면 해외 주재원의 자격을 얻게 된 | 등도 수행하게 되며, 이 과정을 마치면 해외 주재원의 자격을 얻게 된 | 0.7541	True
    --------------------------------------------------------------------------------
    [20500/100000] Train loss: 0.00619, Valid loss: 1.13566, Elapsed_time: 7963.53043
    Current_accuracy : 81.000, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    는 이번 방은 안중근 의사 관련 레포트 우수 대학생 13명, 전국 중 | 는 이번 방은 안중근 의사 관련 레포트 우수 대학생 13명, 전국 중 | 0.3111	True
    7호 객차에 앉아 졸고 있었는데 갑자기 주위가 시끄러워 눈을 떠 보 | 7호 객차에 앉아 졸고 있었는데 갑자기 주위가 시끄러워 눈을 떠 보 | 0.0026	True
    오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 0.8552	True
    으며 3억원의 최종 종착지가 현 전 의원이라는 진술도 신하고 있다. | 으며 3억원의 최종 종착지가 현 전 의원이라는 진술도 신하고 있다. | 0.2137	True
    한국광해관리공단(이사장 권인)은 국내뿐만 아니라 세계 각 국별로 | 한국광해관리공단(이사장 권인)은 국내뿐만 아니라 세계 각 국별로 | 0.7382	True
    --------------------------------------------------------------------------------
    [21000/100000] Train loss: 0.00547, Valid loss: 1.20594, Elapsed_time: 8157.03760
    Current_accuracy : 81.100, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    단축에 따른 중소기업의 부담을 덜어주기 위해지원사업을 펴고 있으 | 단축에 따른 중소기업의 부담을 덜어주기 위해지원사업을 펴고 있으 | 0.7372	True
    고생할 것으로 우려해 딸에게 극약을 먹여 숨지게 한 의로 구속기 | 고생할 것으로 우려해 딸에게 극약을 먹여 숨지게 한 의로 구속기 | 0.8310	True
    "며 "600억원에 이르는 외자 유치를 한다고까지 부풀려져 황당하다" | "며 "600억원에 이르는 외자 유치를 한다고까지 부풀려져 황당하다" | 0.7186	True
    의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 의 정확한 8매량은 모르지만 10일 4르한 있매의 신매은 총분은 것고  | 0.0000	False
    서울우유가 치열한 유업체들간 경쟁시장에 '행복' 발을 들고 나 | 서울우유가 치열한 유업체들간 경쟁시장에 '행복' 발을 들고 나 | 0.6354	True
    --------------------------------------------------------------------------------
    [21500/100000] Train loss: 0.02788, Valid loss: 1.16528, Elapsed_time: 8351.47844
    Current_accuracy : 80.900, Current_norm_ED  : 0.91
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    구하기 위해 신태용 감독이 긴급 투입지만,첫 경기던 이란전에서 | 구하기 위해 신태용 감독이 긴급 투입지 경약 3장이씨가 청 | 0.0001	False
    국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 0.6854	True
    사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 0.5769	True
    00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 0.4644	True
    충격패를 당하며 조 2위 자리가 위태로워지기도 했습니다.시리아와 | 충격패를 당하며 조 2위 자리가 위태로워지기도 했습니다.시리아와 | 0.4512	True
    --------------------------------------------------------------------------------
    [22000/100000] Train loss: 0.00737, Valid loss: 1.36967, Elapsed_time: 8546.26349
    Current_accuracy : 77.600, Current_norm_ED  : 0.90
    Best_accuracy    : 82.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 0.7079	True
    다. 이 관계자는 이어 "전입학 당시 오 교사가 전입학서류를 담당 교 | 다. 이 관계자는 이어 "전입학 당시 오 교사가 전입학서류를 담당 교 | 0.7096	True
    표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 0.7432	True
    문위원회'를 구성해 성 평등에 관한 전반적인 학교 정책을 논의할 예 | 문위원회'를 구성해 성 평등에 관한 전반적인 학교 정책을 논의할 예 | 0.2049	True
    영문 제목)'는 한국에서 1천100만명을 모은 '브라더후드('태극기 휘 | 영문 제목)'는 한편에서 1천000만만을 모은 '브라더 모('태로 휘 | 0.0000	False
    --------------------------------------------------------------------------------
    [22500/100000] Train loss: 0.03377, Valid loss: 1.13615, Elapsed_time: 8739.33978
    Current_accuracy : 82.700, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    "나흘 간의 설 연휴가 시작습니다. 연휴 첫 날, 잘 보내고 계신가요? | "나흘 간의 설 연휴가 시작습니다. 연휴 첫고 7기 내고 요요습 | 0.0003	False
    받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 0.8676	True
    두확인한 결과, 김우중 전 회장이 출국한 이후 어느 쪽 여권으로도 귀 | 두확인한 결과, 김우중 전 회장이 출국한 이후 어느 쪽 여권으로도 귀 | 0.6425	True
    로 삭감는데도 정파적으로 몰고가는 것은 아집과 독선"이라고 말했 | 로 삭감는데도 정파적으로 몰고가는 것은 아집과 독선"이라고 말했 | 0.8152	True
    록 조심하야습니다. 아침 기온은 서울 영하 5도, 대구는 영하 3도 | 에 대심하습니다. 아침 검사장 의회 3산이 만하는 것으 스도으로 국장 | 0.0000	False
    --------------------------------------------------------------------------------
    [23000/100000] Train loss: 0.00522, Valid loss: 1.10997, Elapsed_time: 8932.71278
    Current_accuracy : 81.900, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 0.8753	True
    한국수자원공사의 물관리에 구멍이 뚫다. 두달새 같은 지점에서 유 | 한국수자원공사의 물관리에 구멍이 뚫다. 두달새 같은 지점에서 유 | 0.5274	True
    안부 피해자들이 보다 적극적으로 자신의 명예와 존엄성을 지킬 수 | 안후 피상자들이 보신 적다. 당실의 보리의에 존계성 존엄적을 국킬 수 | 0.0000	False
    의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 0.6461	True
    있었으나 2003년부터 학부모의 실제 거주지는 서울 강남구 개포동 | 있었으나 2003년부터 학부모의 실제 거주지는 서울 강남구 개포동 | 0.5827	True
    --------------------------------------------------------------------------------
    [23500/100000] Train loss: 0.00443, Valid loss: 1.14809, Elapsed_time: 9125.34607
    Current_accuracy : 81.400, Current_norm_ED  : 0.91
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 0.8085	True
    스럽고도 아 역사를 보고 듣고 느끼며 생생히 깨달을 수 있는 기회 | 스럽고도 아 역사를 보고 듣고 느끼며 생생히 깨달을 수 있는 기회 | 0.3759	True
    둥 번개가 치습니다. 비는 밤에 서쪽 지역부터 차차 그치습니다. | 둥 번개가 치습니다. 비는 밤에 서쪽 지역부터 차차 그치습니다. | 0.8575	True
    로 알려다. 법무부 관계자는 "c검사가 제출한 사표는 대검을 통해 | 로 알려다. 법무부 관계자는 "c검사가 제출한 사표는 대검을 통해 | 0.8601	True
    금감독원이 금서비스개선국을 신설하고 외환업무실을 외환감독 | 금감독동 이라 지지와 인문이는 실어 사는에 다. 문으로 국으로  | 0.0000	False
    --------------------------------------------------------------------------------
    [24000/100000] Train loss: 0.00673, Valid loss: 1.13694, Elapsed_time: 9318.74102
    Current_accuracy : 81.800, Current_norm_ED  : 0.91
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 0.6533	True
    심신에 활기를 준다. 슈스버그 블랑 드 누아는 익은 복아 살구 딸 | 심신에 활기를 준다. 슈스버그 블랑 드 누아는 익은 복아 살구 딸 | 0.7807	True
    약인 '모자이크프로트사업'과 '인공습지조성사업' 관련 예산이 포 | 약인 '모자이크프로트자지'는 '공공 조사업'공지''지사업 지원 | 0.0000	False
    생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 0.8765	True
    받고 있다. 권 수석은 전화 통화를 인정했고 "일언지하에 거절했다"고 | 받고 있다. 권 수석은 전화 통화를 인정했고 "일언지하에 거절했다"고 | 0.8839	True
    --------------------------------------------------------------------------------
    [24500/100000] Train loss: 0.00403, Valid loss: 1.15511, Elapsed_time: 9513.30068
    Current_accuracy : 81.100, Current_norm_ED  : 0.91
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 받고 직장에서 직위해제 다는 가족들의 말에따라 숨진 이씨가 신변 | 0.9064	True
    꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 0.5300	True
    조되는 과정에서 높은 파도로 많은 물을마 의식불명 상태로 삼척의 | 조되는 과정에서 높은 파도로 많은 물을마 의식불명 상태로 삼척의 | 0.8326	True
    내일 아침 기온은 서울 12도, 광주 14도로 예상니다. 낮 기온은 서 | 내일 아침 기온은 서울 12도, 광주 14도로 예상니다. 낮 기온은 서 | 0.7886	True
    여학생 중에 (성관계를) 하고 싶은 사람을 골라라는 말을 한 것으로 | 여학생 답에 대성관계를) 했다. 하는 사은 의로 합 있으는 것을 다고, | 0.0000	False
    --------------------------------------------------------------------------------
    [25000/100000] Train loss: 0.00378, Valid loss: 1.19450, Elapsed_time: 9706.32981
    Current_accuracy : 80.400, Current_norm_ED  : 0.91
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    개국 8개 법인에 파견하기로 했다. 이외에도 대한통운은 다양한 사내 | 개국 8개 법장이 사행상 위견하다는 대선통성은 보양대로부 대한 | 0.0000	False
    계는 완전히 격폐 것이며 그 어떤 내왕(왕래)도, 접촉도 이뤄지지 | 으는 비전 없람이 높기로 기 적으로 4명이 새태다는 반반을 전입하 | 0.0000	False
    금실무협의회 이후 십수년만에 부활되는 것이다. 하지만 재정부가 | 금실무협의회 이후 십수년만에 부활되는 것이다. 하지만 재정부가 | 0.9073	True
    제보다 5,6도 가량 낮습니다. 현재 대부분 해상으로 풍랑 특보가 발 | 제보다 5,6도 가량 낮습니다. 현재 대부분 해상으로 풍랑 특보가 발 | 0.8315	True
    생활고에 시달리다 남편과 동반자살을 하기로 한 뒤 혼자남는 딸이 | 생활고에 시달리다 남편과 동반자살을 하기로 한 뒤 혼자남는 딸이 | 0.1374	True
    --------------------------------------------------------------------------------
    [25500/100000] Train loss: 0.00385, Valid loss: 1.20558, Elapsed_time: 9899.20051
    Current_accuracy : 80.600, Current_norm_ED  : 0.91
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    로 수사를 확대하고 있다. 부산지검 공안부(부장검사 이태승)는 3억 | 로 수사를 확대하고 있다. 부산지검 공안부(부장검사 이태승)는 3억 | 0.8331	True
    서울시 교육청이 답안대리 작성과 관련해 서울 동부지검에 수사를 | 서울시 교육청이 답안대리 작성과 관련해 서울 동부지검에 수사를 | 0.8032	True
    감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 0.5055	True
    우포 으 명소 가꾸기, 산청 한방휴양체험특화도시 조성, 합천 대 | 우포 으 명소 가꾸기, 산청 한방휴양체험특화도시 조성, 합천 대 | 0.7205	True
    어에서 주로 볼 수 있는 패으로 타이어 좌우로 이 패여 있는 모양 | 어에서 주로 볼 수 있는 패으로 타이어 좌우로 이 패여 있는 모양 | 0.8642	True
    --------------------------------------------------------------------------------
    [26000/100000] Train loss: 0.07825, Valid loss: 1.18061, Elapsed_time: 10091.89475
    Current_accuracy : 80.600, Current_norm_ED  : 0.91
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 러 다른 정관계 등적인사 등이 연리는 연루는지 추궁다. | 0.0000	False
    영국 폭동 와중에 페이스북에 선동 글을 올린 젊은이들에게 중형이 | 영국 폭동 중중 예산이 페우 마을 선인하는 불북에게 중제이)에 | 0.0000	False
    법률'에 제11조의 3을 신설, 일본군 위안부 피해자들이 명예손이나 | 법률'에 제11조의 3을 신설, 일본군 위안부 피해자들이 명예손이나 | 0.5469	True
    전혀 없었다"고 말했다. 앞서 프랑스 일간 리베라시은 로르 회장의 | 전혀라더라 "이말 고 있간 간리는 일간 리시시리들을 마르'과 장 | 0.0000	False
    단축에 따른 중소기업의 부담을 덜어주기 위해지원사업을 펴고 있으 | 단축에 따른 중소기업의 부담을 덜어주기 위해지원사업을 펴고 있으 | 0.6643	True
    --------------------------------------------------------------------------------
    [26500/100000] Train loss: 0.00641, Valid loss: 1.10895, Elapsed_time: 10283.56220
    Current_accuracy : 82.200, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 0.6552	True
    정이다. 은행권역에서는 은행감독국이 신설되고 외환업무실이 외환 | 정이다. 은행권역에서는 은행감독국이 신설되고 외환업무실이 외환 | 0.8957	True
    신한카드는 5일 경기도 성남시 차의과학대학교에서 이재우 사장과 | 신한카드는 5일 경기도 성남시 차의과학대학교에서 이재우 사장과 | 0.8936	True
    "3.1절인 오늘은 어제와 달리 맑은 하늘이 드러나 있지만, 바람이 상 | "3.1절인 오늘은 어제와 달리 맑은 하늘이 드러나 있지만, 바람이 상 | 0.6508	True
    있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 0.7645	True
    --------------------------------------------------------------------------------
    [27000/100000] Train loss: 0.00420, Valid loss: 1.14062, Elapsed_time: 10475.36692
    Current_accuracy : 82.000, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 건섭 금투서비스국장이 각각 승진 예정이다고 또 신한은행 감사로 | 0.1219	False
    지거나 이러한 수분이 제대로 배출되지 못하면 차가 미끄러지게 된다 | 지거나 바러은 공자이야 대리 불치대면 배출리로에 수가 뒤 되지 | 0.0000	False
    찾는 이들이 많다. 하지만 여름 보양식에 와인을 곁들이면 더욱 좋다 | 주을 곁할 기도에 대신적이 있다"며 인응 아양없을 징러할 것양와 한다 | 0.0000	False
    비 부담을 덜어주기 위해 주40시간제를앞당겨 시행하면서 정규직을 | 비 부담을 덜어주기 위해 주40시간제를앞당겨 시행하면서 정규직을 | 0.8933	True
    해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 0.8883	True
    --------------------------------------------------------------------------------
    [27500/100000] Train loss: 0.00449, Valid loss: 1.13261, Elapsed_time: 10668.85763
    Current_accuracy : 81.200, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    단축에 따른 중소기업의 부담을 덜어주기 위해지원사업을 펴고 있으 | 단축에 따른 중소기업의 부담을 덜어주기 위해지원사업을 펴고 있으 | 0.8111	True
    사진을 위주로, 최근 일본 개봉 스줄이 나온 원 주연의 '우리형' | 사진을 위주로, 최근 일본 개봉 스줄이 나온 원 주연의 '우리형' | 0.6918	True
    나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 0.8152	True
    사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 0.7102	True
    민 1명이 다리에 총상을 입었다. 브라질 경찰은 300여명의 병력을 동 | 민 1명이 다리에 만상을 코었다. 이라 정찰에 병상과 도었이 강도, 만도 | 0.0000	False
    --------------------------------------------------------------------------------
    [28000/100000] Train loss: 0.00371, Valid loss: 1.12082, Elapsed_time: 10859.45740
    Current_accuracy : 82.500, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 0.8387	True
    된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 0.8235	True
    정비기술 향상, 전문인력 양성을 위해 면적 1375m2, 높이 25 미터 | 정비기술 향상, 전문인력 양성을 위해 면적 1375m2, 높이 25 미터 | 0.7567	True
    그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 0.7721	True
    말을 인용, 로르 회장이 한국에서사업을 위해 김 전 회장을 고문역으 | 말을 인용, 로르 회장이 한국에서사업을 위해 김 전 회장을 고문역으 | 0.8266	True
    --------------------------------------------------------------------------------
    [28500/100000] Train loss: 0.00632, Valid loss: 1.20808, Elapsed_time: 11051.77277
    Current_accuracy : 80.900, Current_norm_ED  : 0.91
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    고 거듭 밝다. 김형곤의 동생이 산 암태면 벌목도는 신안군내 753 | 고 거듭 밝다. 김형곤의 동생이 산 암태면 벌목도는 신안군내 753 | 0.6842	True
    의할 예정이라고 8일 밝다. 이 의원이 밝 개정안 내용은 '일제하 | 사할 예정이 고 시국의 안 전책 및 의국이 제정개 이용 민목을 목목하 | 0.0000	False
    문제 해결과 한반도 평화를 일구도록 노력하다고 약속했습니다. 지 | 문제 해결과 한반도 평화를 일구도록 노력하다고 약속했습니다. 지 | 0.8964	True
    입니다.                      | 입다.                       | 0.3637	False
    사진을 위주로, 최근 일본 개봉 스줄이 나온 원 주연의 '우리형' | 조화를 찾동 등 비자를 건벌나 나주 조직의 페리엔'으로 한소나 | 0.0000	False
    --------------------------------------------------------------------------------
    [29000/100000] Train loss: 0.00461, Valid loss: 1.11223, Elapsed_time: 11243.55349
    Current_accuracy : 82.000, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    로 가는 중심가치를 '행복'으로 설정한 것이다. 송용헌 서울우유협동 | 로 가는 중심가치를 '행복'으로 설정한 것이다. 송용헌 서울우유협동 | 0.4937	True
    차량에 많이 쓰인다. 반면 래그형 은 트럭이나 버스에 장착되는 타이 | 차량에 많이 쓰인다. 반면 래그형 은 트럭이나 버스에 장착되는 타이 | 0.8366	True
    선고다. 글랜드 체스터 지방법원이 페이스북에 폭동을 선동하는 | 선고다. 글랜드 체스터 지방법원이 페이스북에 폭동을 선동하는 | 0.8929	True
    명숙 전 총리 사건처럼 정의가 거짓을 이긴다는 사실을 반드시 증명 | 상숙 전 모리 처리처럼 정거가 거짓하  정도 부반반을 반반했 것을  | 0.0000	False
    무가 분리되면서 은행, 보험, 금투자, 중소서민 등 각 권역별로 감독 | 무가 분리되면서 은행, 보험, 금투자, 중소서민 등 각 권역별로 감독 | 0.8557	True
    --------------------------------------------------------------------------------
    [29500/100000] Train loss: 0.00319, Valid loss: 1.14408, Elapsed_time: 11434.31269
    Current_accuracy : 82.100, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    날인 내일은 대부분 지방이 맑지만 제주도는 오후 한때 비나 눈이 | 날인 내일은 대부분 지방이 맑지만 제주도는 오후 한때 비나 눈이 | 0.7768	True
    국으로 확대 개편하는 등 조직을 대폭 확대한다. 특히 감독과 검사업 | 국으로 확대 개편하는 등 조직을 대폭 확대한다. 특히 감독과 검사업 | 0.8553	True
    른 자료를 사용했기 때문으로 밝혀다. 감사원은 기재부장관에게 " | 른 주료를 사기했 기 문문 보고들들 유사들은 기재관게 기기에 | 0.0000	False
    중소기업에 대해 근로시간 단축 지원금을 주고 있는데도 울산지역 | 중소기업에 대해 근로시간 단축 지원금을 주고 있는데도 울산지역 | 0.8954	True
    어지기를 바란다"고 말했다. 한편, 신한카드는 이날 차병원에 소아뇌 | 어지기를 바란다"고 말했다. 한편, 신한카드는 이날 차병원에 소아뇌 | 0.0788	True
    --------------------------------------------------------------------------------
    [30000/100000] Train loss: 0.00290, Valid loss: 1.12802, Elapsed_time: 11626.30820
    Current_accuracy : 81.800, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    정비기술 향상, 전문인력 양성을 위해 면적 1375m2, 높이 25 미터 | 정비기술 향상, 전문인력 양성을 위해 면적 1375m2, 높이 25 미터 | 0.7776	True
    고 있다. 야당과 시민사회단체들은 중형을 선고한 것은 지나치다고 | 고 있다. 야당과 시민사회단체들은 중형을 선고한 것은 지나치다고 | 0.4967	True
    성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 0.8449	True
    주소로돼 있어야 하고 실제 거주해야 한다고 교육청 관계자가 전했 | 주소로돼 있어야 하고 실제 거주해야 한다고 교육청 관계자가 전했 | 0.8542	True
    장경기록문화 테마파크 조성, 통영 국제음악당 건립, 김해 중소기업 | 한 모생식을 선기기 인해대 민주식의 부선전 전반한 반판의와 정화 | 0.0000	False
    --------------------------------------------------------------------------------
    [30500/100000] Train loss: 0.00271, Valid loss: 1.18611, Elapsed_time: 11818.81359
    Current_accuracy : 81.000, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    한인권법을 놓고 하게 맞서는 동안 북한은 이같은 움직임에 대한 | 한인권법을 놓고 하게 맞서는 동안 북한은 이같은 움직임에 대한 | 0.8610	True
    과 유럽 재정위기 등 불안요인이 크고 대내적으로는 물가불안, 가계 | 과 유럽 재정위기 등 불안요인이 크고 대내적으로는 물가불안, 가계 | 0.6790	True
    게 불며 쌀쌀하습니다. 주말에는 대체로 맑은 날씨가 예상니다. | 게 불며 쌀쌀하습니다. 주말에는 대체로 맑은 날씨가 예상니다. | 0.9283	True
    각 중단하고 박 원내대표에게 의가 있다면 당당히 기소하라"며 "한 | 각 중단하고 박 원내대표에게 의가 있다면 당당히 기소하라"며 "한 | 0.4227	True
    받고 있다. 권 수석은 전화 통화를 인정했고 "일언지하에 거절했다"고 | 받고 있다. 권 수석은 전화 통화를 인정했고 "일언지하에 거절했다"고 | 0.9156	True
    --------------------------------------------------------------------------------
    [31000/100000] Train loss: 0.05844, Valid loss: 1.12698, Elapsed_time: 12011.41952
    Current_accuracy : 80.600, Current_norm_ED  : 0.92
    Best_accuracy    : 82.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    한진중공업 김주익 열사와 두산중공업 배달호 열사 등 수많은 동지들 | 한진중공업 김주익 열사와 두산중공업 배달호 열사 등 수많은 동지들 | 0.7573	True
    전혀 없었다"고 말했다. 앞서 프랑스 일간 리베라시은 로르 회장의 | 전혀 없었다"고 말했다. 앞서 프랑스 일간 리베라시은 로르 회장의 | 0.1948	True
    전입학시다는 일부 언론의 보도는사실무근이다"고 해명한 것으로 | 전입학시다는 일부 언론의 보도는사실무근이다"고 해명한 것으로 | 0.6064	True
    여학생 중에 (성관계를) 하고 싶은 사람을 골라라는 말을 한 것으로 | 여학생 중에 (성관계를) 하고 싶은 사람을 골라라는 말을 한 것으로 | 0.6753	True
    와의 동일여부 등을 확인할 예정이다.      | 와의 동일여부 등을 확인할 예정이다.      | 0.9128	True
    --------------------------------------------------------------------------------
    [31500/100000] Train loss: 0.00552, Valid loss: 1.09446, Elapsed_time: 12204.22201
    Current_accuracy : 83.000, Current_norm_ED  : 0.92
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 0.9342	True
    년 가까이 끌어온 북한인권법을 이번 임시국회에서 반드시 처리하 | 년 가까이 끌어온 북한인권법을 이어 임시회회이 반리하 시도 | 0.0067	False
    체 호워드 리그의 드루 닐슨은 "징역 4년형은 흉기로 상해를 입 | 체 호워드 리그의 드루 닐슨은 "징역 4년형은 흉기로 상해를 입 | 0.7212	True
    서 주행을 할 수 있도록 한다. 또한 블록패의 트레드는 바퀴 압력에 | 서 주행을 할 수 있도록 한다. 또한 블록패의 트레드는 바퀴 압력에 | 0.8241	True
    가 참석하는 '거시정책실무협의회'를 월 1회 운영할 계획이라고 밝 | 가 참석하는 '거시정책실무협의회'를 월 1회 운영할 계획이라고 밝 | 0.8810	True
    --------------------------------------------------------------------------------
    [32000/100000] Train loss: 0.00394, Valid loss: 1.10852, Elapsed_time: 12398.67160
    Current_accuracy : 82.700, Current_norm_ED  : 0.92
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 비와 낙동강 사업에 따른 오염 토양환경 안정성 구축사업비도 대상이 | 0.7114	True
    자심문(영장실질심사)은 29일 오전 서울중앙지법에서 열린다. 앞서 | 자심문(영장실질심사)은 29일 오전 서울중앙지법에서 열린다. 앞서 | 0.8986	True
    급을 받기 때문이다. 하지만 감사원 감사결과 이는 평가기준과 다 | 급을 받기 때문이다. 하지만 감사원 감사결과 이는 평가기준과 다 | 0.9406	True
    들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 0.8150	True
    물의를 빚어 감찰 대상에 올던 c검사가 이날 사표를 제출했다고 밝 | 물의를 빚어 감찰 대상에 올던 c검사가 이날 사표를 제출했다고 밝 | 0.8989	True
    --------------------------------------------------------------------------------
    [32500/100000] Train loss: 0.00310, Valid loss: 1.11792, Elapsed_time: 12590.46177
    Current_accuracy : 81.800, Current_norm_ED  : 0.92
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 0.8952	True
    니다. 검찰은 유죄 확정 시 범죄수익을 차질 없이 추징하기 위해 재산 | 니다. 검찰은 유죄 확정 시 범죄수익을 차질 없이 추징하기 위해 재산 | 0.6096	True
    권 수익 전액을 일본 유니세프에 기부하기로 했다. 원의 매니저 장 | 권 수익 전액을 일본 유니세프에 기부하기로 했다. 원의 매니저 장 | 0.3556	True
    해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 0.9107	True
    입을 비롯한 다양한 의혹이제기에 따라 이달 20일부터 c검사에 대 | 입을 비롯한 다양한 의혹이제기에 따라 이달 20일부터 c검사에 대 | 0.6280	True
    --------------------------------------------------------------------------------
    [33000/100000] Train loss: 0.00514, Valid loss: 1.10580, Elapsed_time: 12784.51256
    Current_accuracy : 81.700, Current_norm_ED  : 0.92
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    입장을 발표하고, 사회적으로 손가락질받을 일이 학교에서 발생해 안 | 입장을 발표하고, 사회적으로 손가락질받을 일이 학교에서 발생해 안 | 0.7040	True
    행복한 환경 속에서 자라야 한다는 생각은 '밀크마스터'(milk maste | 행복한 환경 속에서 자라야 한다는 생각은 '밀크마스터'(milk maste | 0.4945	True
    은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 은 "세계 6위의 원전 보유국가이자 원전 수출국가로서선진국 수준의 | 0.6924	True
    니, 33년 만에 카타르에 지면서 8차전은 도하 참사로 기록습니다. | 니, 33년부 부타 오타 지지 10면 도록 오하고 있하고 말하. 사기. | 0.0000	False
    비스실이 여신전문국으로 승격돼 저축은행 감독국, 상호금국과 함 | 비스실이 여신전문국으로 승격돼 저축은행 감독국, 상호금국과 함 | 0.7782	True
    --------------------------------------------------------------------------------
    [33500/100000] Train loss: 0.00341, Valid loss: 1.10860, Elapsed_time: 12978.19397
    Current_accuracy : 82.200, Current_norm_ED  : 0.92
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 0.7883	True
    가야한다"며 노동계의 위상 강화 필요성을 강조했다. 그러나 민주노 | 가야한다. 남편과 동상대화 입제 "활명 민주노고 있다. 또 노동당은 강 | 0.0000	False
    다. 서울우유는 지난 11일 75주년 기념식에서 '행복한 젖소, 행복한 | 다. 서울우유는 지난 11일 75주년 기념식에서 '행복한 젖소, 행복한 | 0.6001	True
    부채 문제 등 취약요인에 대한 적극적인 대응이 필요한 시점이라는 | 부채 문제 등 취약요인에 대한 적극적인 대응이 필요한 시점이라는 | 0.8327	True
    이후 30년간 이 발전을 전세계에 알리는 계기가  것이라고 전망 | 이후 30년간 이 발전을 전세계에 알리는 계기가  것이라고 전망 | 0.7188	True
    --------------------------------------------------------------------------------
    [34000/100000] Train loss: 0.00344, Valid loss: 1.22210, Elapsed_time: 13171.00683
    Current_accuracy : 80.200, Current_norm_ED  : 0.91
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    정책 우선순위를 두는 가운데 고용회복이 지속 수 있는 방향으로 | 정책 우선순위를 두는 가운데 고용회복이 지속 수 있는 방향으로 | 0.8411	True
    나으며 한국노총과 민주노총 간부들도 회의장 안팎에서 진행상황 | 나다며 한국노총과 민주노총 간부들도 회의장 안팎에서 진행상황니다. | 0.0002	False
    임직원들로부터 높은 인기를 얻고 있다. 영어, 중국어, 일본어 외에도 | 임직원들로부터 높은 인기를 얻고 있다. 영어, 중국어, 일본어 외에도 | 0.6818	True
    에서 개봉한다. 일본 산이스포츠는 21일 "'마이 브라더('우리형'의 | 에서 개봉한다. 일본 산이스포츠는 21일 "'마이 브라더('우리형'의 | 0.4019	True
    주소로돼 있어야 하고 실제 거주해야 한다고 교육청 관계자가 전했 | 주소로돼 있어야 하고 실제 거주해야 한다고 교육청 관계자가 전했 | 0.6341	True
    --------------------------------------------------------------------------------
    [34500/100000] Train loss: 0.00294, Valid loss: 1.14722, Elapsed_time: 13361.81371
    Current_accuracy : 81.400, Current_norm_ED  : 0.92
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    어에서 주로 볼 수 있는 패으로 타이어 좌우로 이 패여 있는 모양 | 어에서 주로 볼 수 있는 패으로 타이어 좌우로 이 패여 있는 모양 | 0.8884	True
    다.                        | 다.                        | 0.9942	True
    다. 문 대통령은 오늘 대한노인회 회장단과 가진 오찬에서 과거처럼 | 다. 문 대통령은 오늘 대한노인회 회장단과 가진 오찬에서 과거처럼 | 0.8994	True
    한국수자원공사의 물관리에 구멍이 뚫다. 두달새 같은 지점에서 유 | 한국수자원공사의 물관리에 구멍이 뚫다. 두달새 같은 지점에서 유 | 0.6526	True
    률상담이나 소송대리 등을 지원'하는 개정법률안을 8월 셋째 주에 발 | 률상담이나 소송대리 등을 지원'하는 개정법률안을 8월 셋째 주에 발 | 0.7505	True
    --------------------------------------------------------------------------------
    [35000/100000] Train loss: 0.00254, Valid loss: 1.30270, Elapsed_time: 13553.97894
    Current_accuracy : 78.200, Current_norm_ED  : 0.90
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    방치해 사고를 자초했다는 지적을 받다. 지난달 30일 새벽 낙동강 | 치치가 생희을 자자을 받기로 이지해 설 수장의 사소을 새한 것 | 0.0000	False
    사에게 넘겨준것으로 알고 있다. 원래 학생과 학부모는 함께 전입학 | 정부에 넘르들의 로준으로 것으로 있다. 과래대학 직과 대부가 함께 학 | 0.0000	False
    는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 0.7948	True
    법사위에 계류돼 있는 법안에 북한에 대한 인도적인 지원 내용이 들 | 법사위에 계류돼 있는 법안에 북한에 대한 인도적인 지원 내용이 들 | 0.9416	True
    를 끌어올려 15억여원의 차익을 챙긴 의를 받고 있다. 2010년 2월 | 를 끌어올려 15억여원의 차익을 챙긴 의를 받고 있다. 2010년 2월 | 0.7353	True
    --------------------------------------------------------------------------------
    [35500/100000] Train loss: 0.04069, Valid loss: 1.24786, Elapsed_time: 13745.99828
    Current_accuracy : 80.600, Current_norm_ED  : 0.91
    Best_accuracy    : 83.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
     정도다. 언제나 같이 있으면 재미있고 유쾌한 사람"이라고 말했다. |  정도다. 언제나 같이 있으면 재미있고 유쾌한 사람"이라고 말했다. | 0.8071	True
    식이 열리는 저녁에는 체감온도가 영하 10도 안팎으로 떨어질 것으 | 식이 열리는 저녁에는 체감도안에는 하는 10도의도를 보고으로 있지 | 0.0019	False
    해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 0.8339	True
    청구했다. 잦은 말바꾸기로 증거인멸 등의 우려가 있는 조씨를 그냥 | 청구했다. 잦은 말바꾸기로 증거인멸 등의 우려가 있는 조씨를 그냥 | 0.5440	True
    끝에 확정습니다.감독 경질까지 겪어야 했던 대표팀의 상처 많은 | 끝에 확정습니다.감독 경질까지 겪어야 했던 대표팀의 상처 많은 | 0.6612	True
    --------------------------------------------------------------------------------
    [36000/100000] Train loss: 0.00402, Valid loss: 1.11786, Elapsed_time: 13937.10141
    Current_accuracy : 83.100, Current_norm_ED  : 0.92
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    동 자택과 1억 원짜리 수표 30장 등 모두 58억 원 규몹니다. cj그룹 | 동 자택과 1억 원짜리 수표 30장 등 모두 58억 원 규몹니다. cj그룹 | 0.4383	True
    제하는 것은 표현의 자유를 가로막을 뿐이며 정부가 반정부 시위 때 | 제하는 것은 표현의 자유를 가로막을 뿐이며 정부가 반정부 시위 때 | 0.8863	True
    00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 0.8460	True
    이다. 소음이 많은 것이 단점으로 꼽히기는 하지만 전후 방향의 강력 | 이다. 기로은 지시자으로 은인'으도 지리' 단지 방을 후방있 | 0.0000	False
    해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 0.9131	True
    --------------------------------------------------------------------------------
    [36500/100000] Train loss: 0.00270, Valid loss: 1.14217, Elapsed_time: 14128.12602
    Current_accuracy : 82.800, Current_norm_ED  : 0.92
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    남북갈등으로까지 확산 조짐마저 보이고 있다.  | 남북갈등으로까지 확산 조짐마저 보이고 있다.  | 0.8589	True
    수의사로서 젖소의 스트레스와 건강을 관리하고 있다. 한편, 국내 우 | 수의사로서 젖소의 스트레스와 건강을 관리하고 있다. 한편, 국내 우 | 0.8954	True
    안부 피해자들이 보다 적극적으로 자신의 명예와 존엄성을 지킬 수 | 안부 피해자들이 보다 적극적으로 자신의 명예와 존엄성을 지킬 수 | 0.9157	True
    하는 점에 비 b고교의 전입학 시스템에 중대 허점이 있었던 사실도 | 하는 점에 비 b고 있의 학입에서 분점  사해 아도 점사하고 | 0.0000	False
    의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 0.9415	True
    --------------------------------------------------------------------------------
    [37000/100000] Train loss: 0.00266, Valid loss: 1.15334, Elapsed_time: 14317.96814
    Current_accuracy : 82.200, Current_norm_ED  : 0.92
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    상이라 모욕했다. 이에 대해 일본군 위안부 피해자인 이용수 할머니 | 상이라 모욕했다. 이에 대해 일본군 위안부 피해자인 이용수 할머니 | 0.9349	True
    위원장 등 간부들과 간담회를 갖고 노동현안 등을 논의했다. 문 후보 | 위원장 등간 관시를 불제럽 의원장보을 논장했다. 또 대통 | 0.0000	False
    록 조심하야습니다. 아침 기온은 서울 영하 5도, 대구는 영하 3도 | 록 조심하야습니다. 아침 기온은 서울 영하지 도도 구주다는 오는 예 | 0.0006	False
    에 소재파악 수사를 의해 상태"라며 "만약 김 전 회장이 국내에 | 에 소재자악 수사를 의해 상태"라며 "만약 김 전 회장이 국내에 | 0.0146	False
    표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 0.9118	True
    --------------------------------------------------------------------------------
    [37500/100000] Train loss: 0.00354, Valid loss: 1.21269, Elapsed_time: 14509.67220
    Current_accuracy : 81.400, Current_norm_ED  : 0.91
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    차량에 많이 쓰인다. 반면 래그형 은 트럭이나 버스에 장착되는 타이 | 차량에 많이 쓰인다. 반면 래그형 은 트럭이나 버스에 장착되는 타이 | 0.8561	True
    에 입맛이 떨어지기 십상인 여름에 더할 나위 없이 좋다. 초복을 시작 | 에 입맛이 떨어지기 십상인 여름에 더할 나위 없이 좋다. 초복을 시작 | 0.8156	True
    하는 점에 비 b고교의 전입학 시스템에 중대 허점이 있었던 사실도 | 하는 점에 비 b고 있었다 입사들을 올던 허도서 지사에 대해 | 0.0000	False
    무가 분리되면서 은행, 보험, 금투자, 중소서민 등 각 권역별로 감독 | 무가 분리되면서 은행, 보험, 금투자, 중소서민 등 각 권역별로 감독 | 0.8914	True
    한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 한 국방력을 기반으로 남북 대화와 평화를 추구해 가다고 말했습니 | 0.8897	True
    --------------------------------------------------------------------------------
    [38000/100000] Train loss: 0.00297, Valid loss: 1.15389, Elapsed_time: 14699.22193
    Current_accuracy : 82.000, Current_norm_ED  : 0.92
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    공동브리핑에서 "우리 경제가 수출 호조와 고용 개선 등에 힘입어 잠 | 공동브리핑에서 "우리 경제가 수출 호조와 고용 개선 등에 힘입어 잠 | 0.9073	True
    입니다. 다만 제주도는 일요일 오후부터 다시 비가 오기 시작하습 | 입니다. 다만 제주도는 일요일 오후부터 다시 비가 오기 시작하습 | 0.9405	True
    방치해 사고를 자초했다는 지적을 받다. 지난달 30일 새벽 낙동강 | 치치가 원희을 자자했다는 지적을 받다"고 말달해 새벽이 새벽 주당해  | 0.0000	False
    그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 0.8373	True
    된다. 고무에 카본블주1을 섞은 일반타이어는 날씨가 추워질수록 | 된다. 고무에 카본블주1을 섞은 일반타이어는 날씨가 추워질수록 | 0.8747	True
    --------------------------------------------------------------------------------
    [38500/100000] Train loss: 0.00290, Valid loss: 1.21273, Elapsed_time: 14889.48709
    Current_accuracy : 81.300, Current_norm_ED  : 0.91
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    료원으로 옮으나 숨다. 해경과 소방서 등으로 이뤄진 구조대는 | 료원으로 옮으나 숨다. 해경과 소방서 등으로 이뤄진 구조대는 | 0.8972	True
    적하고 있다. 7조원대의 금비리를 저지른 부산저축은행이 살아남 | 적하고 있다. 7조원대의 금비리를 저지른 부산저축은행이 살아남 | 0.8337	True
    아들 c군은 주소지의 학교를 피하기 위해 서울 강동구로 위장전입한 | 아들 c군은 주소지의 학교를 피하기 위해 서울 강동구로 위장전입한 | 0.9511	True
    개국 8개 법인에 파견하기로 했다. 이외에도 대한통운은 다양한 사내 | 개국 8개 법장에 파견하기로 했다. 이외에도 대한다. 밝대해 대내 | 0.0000	False
    서 받은 자료를 바탕으로 책 두권 분량의 탄원서를 만들어 금감독 | 서 받은 자료를 바탕으로 책 두권 분량의 탄원서를 만들어 금감독 | 0.8637	True
    --------------------------------------------------------------------------------
    [39000/100000] Train loss: 0.00253, Valid loss: 1.15729, Elapsed_time: 15080.18098
    Current_accuracy : 82.200, Current_norm_ED  : 0.92
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 성적 언행으로 피해 여학생들이 느을 성적 수치심과 인격적 모멸감 | 0.8893	True
    인하대는 의예과 남학생들의 여학생 집단 성희과 관련해 9일 공식 | 인하대는 의예과 남학생들의 여학생 집단 성희과 관련해 9일 공식 | 0.8890	True
    공조를 강화하기 위해서다. 박재완 기획재정부 장관과 김중수 한국은 | 공조를 강화하기 위해서다. 박재완 기획재정부 장관과 김중수 한국은 | 0.9577	True
    스 사진도 곁들여진다. '우리형'은 일본 uip를 통해 5월 28일 일본 | 쟁  1다도 현들형진'' 형형 구리"은 일본 구음이 높들이 한다" | 0.0000	False
    0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 0.8634	True
    --------------------------------------------------------------------------------
    [39500/100000] Train loss: 0.00202, Valid loss: 1.14440, Elapsed_time: 15269.29692
    Current_accuracy : 82.500, Current_norm_ED  : 0.92
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 의 정확한 판매량은 모르지만 100은 모자 판매한 것매은 보매된 있 | 0.0000	False
    성은 배제할 수 없을것으로 보여 향후 검찰의 수사가 주목된다. | 성은 배제할 수 없을것으로 보여 향후 검찰의 수사가 주목된다. | 0.9433	True
    행복한 환경 속에서 자라야 한다는 생각은 '밀크마스터'(milk maste | 행복한 환경 속에서 자라야 한다는 생각은 '밀크마스터'(milk maste | 0.1751	True
    의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 0.4950	True
    두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 두봉는는 "전전 리액은 전 전 금리에 지태하는 사사이 전달하는 사다. 사에 | 0.0000	False
    --------------------------------------------------------------------------------
    [40000/100000] Train loss: 0.05208, Valid loss: 1.07085, Elapsed_time: 15460.71181
    Current_accuracy : 83.000, Current_norm_ED  : 0.92
    Best_accuracy    : 83.100, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    구조를 재개키로 했다. 한편 이날 좌초된 선박은 부산에서 석회석을 | 공류로 비격이 한 했다"고 말은 공 전 전은 다날 또달대 기태수석 | 0.0000	False
    결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 0.8163	True
    지하철 7호선 방화사건을 수사중인 경기도 광명경찰서는 사건당일 | 지하철 7호선 방화사건을 수사중인 경기도 광명경찰사는 사준당 | 0.1400	False
    표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 표로 광해방지 기술 중 오염토양 복원기술, 광산배수 처리기술을 표 | 0.8857	True
    불러다. 수자원공사는 지난 5월 사고 당시 다시는 사고가 발생하지 | 불러다. 수자원공사는 지난 5월 사고 당시 다시는 사고가 발생하지 | 0.9209	True
    --------------------------------------------------------------------------------
    [40500/100000] Train loss: 0.00331, Valid loss: 1.09148, Elapsed_time: 15650.87692
    Current_accuracy : 83.300, Current_norm_ED  : 0.92
    Best_accuracy    : 83.300, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    록 조심하야습니다. 아침 기온은 서울 영하 5도, 대구는 영하 3도 | 록 조심하야습니다. 아침 기온은 서울 30 5도, 대구는 영하 3도 | 0.0068	False
    사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 0.7874	True
    블록패을 띄고 있다. 자동차가 눈길을 주행할 때 바퀴가 눈을 누르 | 블록패을 띄고 있다. 자동차가 눈길을 주행할 때 바퀴가 눈을 누르 | 0.8171	True
    영장을 청구했으며 긴급체포돼 수감 중인 은 전 위원을 이날 다시 불 | 영장을 청구했으며 긴급체포돼 수감 중인 은 전 위원을 이날 다시 불 | 0.8524	True
    사 사장은 연임이 유력한 것으로 알려다. 공기업 경영평가에서 a등 | 사 사장은 연임로유 등용사 선다. 이다. 이일 간정의 원장 | 0.0000	False
    --------------------------------------------------------------------------------
    [41000/100000] Train loss: 0.00299, Valid loss: 1.13691, Elapsed_time: 15840.55132
    Current_accuracy : 82.100, Current_norm_ED  : 0.92
    Best_accuracy    : 83.300, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    연합뉴스와의 전화 통화를 통해 "최근 변호사인 친동생(김형진 씨)이 | 연합뉴스와의 전화 통화를 통해 "최근 변호사인 친동생(김형진 씨)이 | 0.3622	True
    감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 감이 좋고 시원하게 마실 수 있는 화이트 와인이 좋다. 특히 시원한 스 | 0.6228	True
    이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 0.9230	True
    장을 고발했지만 무의 처리다. 당시 오 교사는 아는 검사를 통해 | 장을 고발했지만 무의 처리다. 당시 오 교사는 아는 검사를 통해 | 0.9243	True
    공음면에서는 지난 23일부터 '청보리 축제'가 한창이다. 올해 8회를 | 공음면에서는 지난 23일부터 '청보리 축제'가 한창이다. 올해 8회를 | 0.8379	True
    --------------------------------------------------------------------------------
    [41500/100000] Train loss: 0.00222, Valid loss: 1.06907, Elapsed_time: 16031.55066
    Current_accuracy : 82.900, Current_norm_ED  : 0.92
    Best_accuracy    : 83.300, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    자들의 명예가 손 우려가 있는 사건들이 늘고 있다"며 "일본군 위 | 자들의 명예가 손 우려가 있는 사건들이 늘고 있다"며 "일본군 위 | 0.9369	True
    감독국으로 확대된다. 또 외은지점의 감독과 검사를 전담하는 별도의 | 감독국으로 확대된다. 또 외은지점의 감독과 검사를 전담하는 별도의 | 0.9497	True
    대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 0.9362	True
    찾는 이들이 많다. 하지만 여름 보양식에 와인을 곁들이면 더욱 좋다 | 주을 확할 기양에 대신적이다. 하지만 이후 양양식 지태을 양할 기양한 한 | 0.0000	False
    송수진입니다."                  | 송수진입니다."                  | 0.9741	True
    --------------------------------------------------------------------------------
    [42000/100000] Train loss: 0.00200, Valid loss: 1.06341, Elapsed_time: 16222.12572
    Current_accuracy : 83.100, Current_norm_ED  : 0.92
    Best_accuracy    : 83.300, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    럴 플랜'를 출시했다. 옥수수를 주원료로 한 배합사료 대신 풀을 먹 | 럴 플랜'를 "아할 문 수료는 건 몰황를 광련되다. 와 한인 때체부 공리 | 0.0000	False
    국으로 확대 개편하는 등 조직을 대폭 확대한다. 특히 감독과 검사업 | 국으로 확대 개편하는 등 조직을 대폭 확대한다. 특히 감독과 검사업 | 0.6657	True
    으로 모실 예정"이라고 말했다. 문 후보는 또 "참여정부는 한계가 있 | 으로 모실 예정"이라고 말했다. 문 후보는 또 "참여정부는 한계가 있 | 0.9489	True
    들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 들이 일사불란하게 밀어붙다"며 "특정 정당의 포"라고 말했다. | 0.9452	True
    했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 0.8392	True
    --------------------------------------------------------------------------------
    [42500/100000] Train loss: 0.00207, Valid loss: 1.12088, Elapsed_time: 16409.54479
    Current_accuracy : 82.500, Current_norm_ED  : 0.92
    Best_accuracy    : 83.300, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 0.8764	True
    회복을 중심으로 거시정책이 운영돼야 한다는 공통인식에 따라 정책 | 회복을 중심으로 거시정책이 운영돼야 한다는 공통인식에 따라 정책 | 0.6293	True
    만 아니라 전후좌우 방향으로 미끄러지는 것을 방지하는 효과가 있다 | 만 아니라 전후좌우 방향으로 미끄러지는 것을 방지하는 효과가 있다 | 0.6241	True
    휴가를 요청해 놓으며 형사대를 이씨의 군부대에 보내 이씨를 경찰 | 휴가를 요청해 놓으며 형사대를 이씨의 군부대에 보내 이씨를 경찰 | 0.8965	True
    마당 등을 선보일 계획이다. 남원에선 5월 6일부터 10일까지 문화체 | 마당 등을 선보일 계획이다. 남원에선 5월일선 전씨에서 선행지까지 문화체 | 0.0000	False
    --------------------------------------------------------------------------------
    [43000/100000] Train loss: 0.00185, Valid loss: 1.03750, Elapsed_time: 16599.62769
    Current_accuracy : 83.400, Current_norm_ED  : 0.92
    Best_accuracy    : 83.400, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    원장, 감사원장에게 보 사실은 인정하고 있으며 이는 고문변호사로 | 원장, 감사원장에게 보 사실은 인정하고 있으며 이는 고문변호사로 | 0.9600	True
    온 영하 6도까지 떨어지는 등 반짝 추위가 이어지습니다. 하지만, | 온 영하 6도까지 떨어지는 등 반짝 추위가 이어지습니다. 하지만, | 0.9458	True
    통해 금감원 검사에서 '강도와 제재수준을 완화해달라'는 취지로 당 | 통해 금감원 검사에서 '강도와 제재수준을 완화해달라'는 취지로 당 | 0.9574	True
    에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 0.9051	True
    조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 0.9155	True
    --------------------------------------------------------------------------------
    [43500/100000] Train loss: 0.00174, Valid loss: 1.06077, Elapsed_time: 16789.24682
    Current_accuracy : 82.900, Current_norm_ED  : 0.92
    Best_accuracy    : 83.400, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    대중공업 등 대기업 협력업체들이 모기업에 따라 불가피하게 근로시 | 대중공업 등 대기업 협력업체들이 모기업에 따라 불가피하게 근로시 | 0.9459	True
    나 있고 미세 먼지 농도도 보통 단계입니다. 하지만, 전남과 제주 지역 | 나 있고 미세 먼지 농도도 보통 단계입니다. 하지만, 전남과 제주 지역 | 0.9418	True
    어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 | 어가 반한 때문에 이 법안 원안 '안을 가지되 이 한다"며 동동 | 0.0000	False
    인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 0.9586	True
    러난 박 전 대통령의 전 재산입니다. 검찰은 수표와 현금 등 40억여 | 러난 박 전 대통령의 전 재산입니다. 검찰은 수표와 현금 등 40억여 | 0.3366	True
    --------------------------------------------------------------------------------
    [44000/100000] Train loss: 0.00170, Valid loss: 1.07111, Elapsed_time: 16978.98678
    Current_accuracy : 83.100, Current_norm_ED  : 0.92
    Best_accuracy    : 83.400, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 자유스럽지 못했지만 노동문제를 너무 기술적으로만 접근했다"며 " | 0.9054	True
    시한 결과다. 경북도와 구미시는 물론 가물막이 아래 준설공사를 담 | 시한 결과다. 경북도와 구미시는 물론 가물막이 아래 준설공사를 담 | 0.4821	True
    정보습니다."                   | 정보습니다."                   | 0.9953	True
    2월 모상을 당했을 때도 부인 정희자씨만 입국했을뿐 김 전 회장은 | 원장 총상을 부상했식 망입한 전문인 자환인을 희박 환희을 당부 보 | 0.0000	False
    여야 원내대표간 합의했던 등록금 부담완화 문제와 저축은행 국정조 | 여야 원내대표간 합의했던 등록금 부담완화 문제와 저축은행 국정조 | 0.8525	True
    --------------------------------------------------------------------------------
    [44500/100000] Train loss: 0.00190, Valid loss: 1.14240, Elapsed_time: 17170.14409
    Current_accuracy : 82.000, Current_norm_ED  : 0.92
    Best_accuracy    : 83.400, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    다"고 벼르고 있다. 민주당은 "한나라당 법안은 보수단체 지원내용만 | 다"고 벼르고 있다. 민주당은 "한나라당 법안은 보수단체 지원내용만 | 0.3551	True
    자원공사측에 건의했다. 하지만 수자원공사는 사고직전 가물막이 시 | 자원공사측에 건의했다. 하지만 수자원공사는 사고직전 가물막이 시 | 0.2527	True
    스 사진도 곁들여진다. '우리형'은 일본 uip를 통해 5월 28일 일본 | 쟁  1진 도 여형진'다. 경우리형형은 일본 u통p를 통해 5월 28일 일본 | 0.0000	False
    블리인 리엄 페브르 그랑크 '레끌로'를 추천한다. 흰 과일, 꽃, 스 | 블리인 리엄 페브르 그랑크 '레끌로'를 추천한다. 흰 과일, 꽃, 스 | 0.6835	True
    소재로 한 '2011 고창 복분자 푸드페스티벌'을 개최해 축제의 계절에 | 소재로 전 해제리 제 상음된 로 마자와 출포해 비 자제 보에에 | 0.0000	False
    --------------------------------------------------------------------------------
    [45000/100000] Train loss: 0.00171, Valid loss: 1.07550, Elapsed_time: 17359.94493
    Current_accuracy : 82.900, Current_norm_ED  : 0.92
    Best_accuracy    : 83.400, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    소다.                       | 소다.                       | 0.9838	True
    파도가 높고 날이 어두워지자 이날 구조작업은 철수하고 17일 오전 | 파도가 높이 날 있어워는 지난 조안 업체가를 했다고 했다. 7 전유 | 0.0000	False
    자를 목격한 사실을 확인했다. 경찰은 이씨의 군부대에 이씨의 특별 | 자를 목격한 사실을 확인했다. 경찰은 이씨의 군부대에 이씨의 특별 | 0.9710	True
    두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 0.9035	True
    호사는 박 전 대통령의 현금 10억여 원도 보관 중입니다. 지금까지 드 | 호사는 박 전 대통령의 현금 10억여 원도 보관 중입니다. 지금까지 드 | 0.9322	True
    --------------------------------------------------------------------------------
    [45500/100000] Train loss: 0.04477, Valid loss: 1.20705, Elapsed_time: 17549.35835
    Current_accuracy : 81.400, Current_norm_ED  : 0.91
    Best_accuracy    : 83.400, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    서울시 교육청 관계자 등은 20일 c군이 b고에 전입학한 작년 3월 가 | 서울시 교육청 관계자 등은 20일 c군이 b고에 전입학한 작년 3월 가 | 0.7143	True
    한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 한편,2006년mbc에 입사한 손정은 아나운서는 현재mbc '뉴스투 | 0.0493	True
    법무부는 21일 담임교사가 아들의 답안지를 대신 작성한 사건으로 | 법무부는 21일 담임교사가 아들의 답안지를 대신 작성한 사건으로 | 0.8515	True
    습니다. 오늘 경북 상주는 낮 기온 25도까지 올고 서울도 22.1도 | 습니다. 오늘 경북 상주는 낮 기온 25도까지 올고 서울도 22.1도 | 0.8394	True
    역으로 지목하고 있는 '다이아드 아일랜드'에 포함돼 있어 이 같은 | 역으로 지목하고 있는 '다이아드 아일랜드'에 포함돼 있어 이 같은 | 0.7895	True
    --------------------------------------------------------------------------------
    [46000/100000] Train loss: 0.00296, Valid loss: 1.02956, Elapsed_time: 17739.18043
    Current_accuracy : 83.000, Current_norm_ED  : 0.92
    Best_accuracy    : 83.400, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 0.8733	True
    서울우유가 치열한 유업체들간 경쟁시장에 '행복' 발을 들고 나 | 서울우유럽 치열한 유업체들간 간쟁시장에 '행행'을 발복 주들가 고 | 0.0000	False
    는 23일 법안심사소위를 열어 정부가 제출한 비정규직법안을 심사해 | 는 23일 법안심사소위를 열어 정부가 제출한 비정규직법안을 심해해 | 0.4657	False
    원의 중간 전달자로 지목된 조기문씨가 돈을 받은 3월 15일과 다음날 | 원의 중간 전달자로 지목된 조기문씨가 돈을 받은 3월 15일과 다음날 | 0.8172	True
    의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 0.7883	True
    --------------------------------------------------------------------------------
    [46500/100000] Train loss: 0.00233, Valid loss: 1.07979, Elapsed_time: 17928.35642
    Current_accuracy : 83.700, Current_norm_ED  : 0.92
    Best_accuracy    : 83.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    공음면에서는 지난 23일부터 '청보리 축제'가 한창이다. 올해 8회를 | 공음면에서는 지난 23일부터 '청보리 축제'가 한창이다. 올해 8회를 | 0.8539	True
    건축학과를 졸업하고 현재중공업분야에서 해외 수출 업무를 전담하 | 건축학과를 졸업하고 현재중공업분야에서 해외 수출 업무를 전담하 | 0.5715	True
    그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 그룹은 최고 와 하지 크다. 밀찰 보동감준을 아아 지면 때세에 투기 | 0.0000	False
    국정원 뇌물 사건을 맡은 유영하 변호사가 보관하고 있습니다. 유 변 | 국정원 뇌물 사건을 맡은 유영하 변호사가 보관하고 있습니다. 유 변 | 0.8897	True
    원장, 감사원장에게 보 사실은 인정하고 있으며 이는 고문변호사로 | 원장, 감사원장에게 보 사실은 인정하고 있으며 이는 고문변호사로 | 0.9368	True
    --------------------------------------------------------------------------------
    [47000/100000] Train loss: 0.00218, Valid loss: 1.05176, Elapsed_time: 18118.62666
    Current_accuracy : 83.500, Current_norm_ED  : 0.92
    Best_accuracy    : 83.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    j그룹 회장에게 그 뜻을 전달했다고 진술했습니다. kbs 뉴스 이현기 | j그룹 회장에게 그 뜻을 전달했다고 진술했습니다. kbs 뉴스 이현기 | 0.9380	True
    해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 0.9468	True
    있었으나 2003년부터 학부모의 실제 거주지는 서울 강남구 개포동 | 있었으나 2003년부터 학부모의 실제 거주지는 서울 강남구 개포동 | 0.9018	True
    타깝다며 인성 교육을 더욱 강화하다고 밝다.학교 측은 이번 성 | 정깝면면 인성 성성을 받욱 하고 계계하게 밝다.학교 측은 이번 성 | 0.0000	False
    전북도를 찾는 관광객에게 오랫동안 기억에 남을 추억거리를 선물할 | 전북도를 찾는 관광객에게 오랫동안 기억에 남을 추억거리를 선물할 | 0.8032	True
    --------------------------------------------------------------------------------
    [47500/100000] Train loss: 0.00177, Valid loss: 1.06915, Elapsed_time: 18307.31949
    Current_accuracy : 83.500, Current_norm_ED  : 0.92
    Best_accuracy    : 83.700, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 된다"고 말했다. 대검 감찰부는 시험답안 대리작성과 관련해 위장전 | 0.9101	True
    니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 니 어떤 남자가 불을 집어 던지는 것이 보다"고 말했다. 경찰은 일 | 0.9653	True
    나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 0.8909	True
    고생할 것으로 우려해 딸에게 극약을 먹여 숨지게 한 의로 구속기 | 고생할 것으로 우려해 딸에게 극약을 먹여 숨지게 한 의로 구속기 | 0.8894	True
    교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 교사는 "2001년 6월 한 교사가 보충수업비 령 등 4가지 의로 교 | 0.9280	True
    --------------------------------------------------------------------------------
    [48000/100000] Train loss: 0.00171, Valid loss: 1.04692, Elapsed_time: 18497.59821
    Current_accuracy : 84.000, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    대구지법 제 12형사부(재판장 김종필 부장판사)는 1일 생활고로 남 | 대구지법 제 12형사부(재판장 김종필 부장판사)는 1일 생활고로 남 | 0.9441	True
    따라 비정규직법안 강행 처리시 24일 오전 8시부터 총파업을 벌이기 | 따라 비정규직법안 강행 처리시 24일 오전 8시부터 총파업을 벌이기 | 0.8806	True
    wonbin'을 열어 팬들과 만난다. 약 5천여명의 팬들이 참석할 예정 | 위해배배 시람이라도 한다. 4들'면 같다"고 명과5사에 따르 요 | 0.0000	False
    며 논란 끝에 예산을 전액 삭감했다. 낙동강특위 활동기간 연장 지원 | 며 논란 끝에 예산을 전액 삭감했다. 낙동강특위 활동기간 연장 지원 | 0.8565	True
    국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 국에서 출품한 190편의 영화가 상영된다. 5월 5일부터는 전주한지문 | 0.8868	True
    --------------------------------------------------------------------------------
    [48500/100000] Train loss: 0.00158, Valid loss: 1.08190, Elapsed_time: 18687.86572
    Current_accuracy : 83.900, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    정이다. 은행권역에서는 은행감독국이 신설되고 외환업무실이 외환 | 정이다. 은행권역에서 은행으로 국설되고 외환업장 실응 민외환 | 0.0000	False
    을 위해폭동 가담자들을 엄벌해야 한다는 의견도 하다. 소 네 | 을 위해폭동 가담자들을 엄벌해야 한다는 의견도 하다. 소 네 | 0.9224	True
    명했다. 답안지 대리작성도 지난해 1학기 중간고사 이전부터 미리 공 | 명했다. 답안지 대리작성도 지난해 1학기 중간고사 이전부터 미리 공 | 0.8764	True
    망각하는 것이며 국민적 평화통일 의지를 손하는 과오를 범하는 것 | 망각하는 것이며 국민적 평화통일 의지를 손하는 과오를 범하는 것 | 0.9167	True
    과 유럽 재정위기 등 불안요인이 크고 대내적으로는 물가불안, 가계 | 과 유럽 재정위기 등 불안요인이 크고 대내적으로는 물가불안, 가계 | 0.8221	True
    --------------------------------------------------------------------------------
    [49000/100000] Train loss: 0.00151, Valid loss: 1.07119, Elapsed_time: 18879.05999
    Current_accuracy : 83.800, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    상태로 이뤄지는 콜드체인시스템을 도입해 2000여 곳의 전용목장에 | 상태로 이뤄지는 콜드체인시스템을 도입해 2000여 곳의 전용목장에 | 0.7808	True
    대구지법 제 12형사부(재판장 김종필 부장판사)는 1일 생활고로 남 | 대구지법 제 12형사부(재판장 김종필 부장판사)는 1일 생활고로 남 | 0.9460	True
    동강 단관로 사고시설 파손우려가 있다고 통보했고, 수회에 걸쳐 | 각나 사건을 설어 조사이 가능성 술사을 가아하고 있다" 사 전인성과 | 0.0000	False
    으며 3억원의 최종 종착지가 현 전 의원이라는 진술도 신하고 있다. | 으며 3억원의 최종 종착지가 현 전 의원이 고 진 전 가획고으도 말 | 0.0001	False
    을 철저하게 하야습니다. 현재 대부분 지역에 맑은 하늘이 드러 | 을 철저하게 하야습니다. 현재 대부분 지역에 맑은 하늘이 드러 | 0.9159	True
    --------------------------------------------------------------------------------
    [49500/100000] Train loss: 0.00145, Valid loss: 1.08183, Elapsed_time: 19069.29481
    Current_accuracy : 83.500, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    통해 금감원 검사에서 '강도와 제재수준을 완화해달라'는 취지로 당 | 통해 금감원 검사에서 '강도와 제재수준을 완화해달라'는 취지로 당 | 0.9589	True
    있다"고 말했다. 또한 안 팀장은 "정부는 일본 정부에 일본군 위안부 | 있다"고 말했다. 또한 안 팀장은 "정부는 일본 정부에 일본군 위안부 | 0.9258	True
    고 있습니다. 화재 예방에 유의하야습니다. 내일은 전국에 구름 | 고 있습니다. 화재 예방에 유의하야습니다. 내일은 전국에 구름 | 0.9659	True
    편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 0.8492	True
    원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 0.9522	True
    --------------------------------------------------------------------------------
    [50000/100000] Train loss: 0.00160, Valid loss: 1.09980, Elapsed_time: 19259.10148
    Current_accuracy : 83.400, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    대중공업 등 대기업 협력업체들이 모기업에 따라 불가피하게 근로시 | 대중공업 등 대기업 협력업체들이 모기업에 따라 불가피하게 근로시 | 0.9563	True
    구조된 구모(59)씨가 숨다. 구씨는 배와 육지를 로프를 연결해 구 | 구조된 구모(59)씨가 숨다. 구씨는 배와 육지를 로프를 연결해 구 | 0.9202	True
    명했다. 답안지 대리작성도 지난해 1학기 중간고사 이전부터 미리 공 | 명했다. 답안지 대리작성도 지난해 1학기 중간고사 이전부터 미리 공 | 0.9378	True
    예상니다. 낮 기온은 서울 3도, 광주 8도, 부산은 10도까지 오르 | 예상니다. 낮 기온은 서울 3도, 광주 8도, 부산은 10도까지 오르 | 0.8725	True
    인 16일 휴대전화 위치 추적을 벌인 결과 현 전 의원과 동선이 일치하 | 는 16일 현대전 위실이 벌들지조 이운 결결 의위  결성장 정책해 위원 | 0.0000	False
    --------------------------------------------------------------------------------
    [50500/100000] Train loss: 0.00138, Valid loss: 1.08292, Elapsed_time: 19451.28911
    Current_accuracy : 83.100, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    조되는 과정에서 높은 파도로 많은 물을마 의식불명 상태로 삼척의 | 조되는 과정에서 높은 파도로 많은 물을마 의식불명 상태로 삼척의 | 0.8540	True
    생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 0.9502	True
    는 "노동계가 기대가 던 데 반해 참여정부가 한계가 있었다는 점을 | 는 "노동계가 기대가 던 데 반해 참여정부가 한계가 있었다는 점을 | 0.9088	True
    4년차 이상 실무자로 6개월 간 현지 문화와 언어를 익히고 시장 조사 | 4년차 이상 실무자로 6개월 간 현지 문화와 언어를 익히고 시장 조사 | 0.6974	True
    단송수관로 누수발생사고도 인재다. 구미시는 5월 16일 문서로 낙 | 단송수관로 누수발생사고도 인재다. 구미시는 5월 16일 문서로 낙 | 0.9554	True
    --------------------------------------------------------------------------------
    [51000/100000] Train loss: 0.00134, Valid loss: 1.10562, Elapsed_time: 19641.70741
    Current_accuracy : 83.300, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 0.9482	True
    고 더 큰 사회 문제를 일으키는 원인이  것"이라며 "결국 그들이 당 | 고 더 큰 사회 문제를 일으키는 원인이  것"이라며 "결국 그들이 당 | 0.7797	True
    의 대표주자로 국내에서도 인기를 누리고 있으며, 각종 와인 평가에 | 의 대표주자로 국내에서도 인기를 누리고 있으며, 각종 와인 평가에 | 0.9640	True
    그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 0.9011	True
    한통운은 최근 어학능력과 국제 감각을 갖춘 '글로벌 인재 풀'을 강화 | 한통운은 최근 어학능력과 국제 감각을 갖춘 '글로벌 인재 풀'을 강화 | 0.9287	True
    --------------------------------------------------------------------------------
    [51500/100000] Train loss: 0.00137, Valid loss: 1.08519, Elapsed_time: 19833.08302
    Current_accuracy : 83.000, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 0.9366	True
    습니다. 서해와 남해안에서는 짙은 안개를 주의하야습니다. 비 | 습니다. 서해에서 1검사 이해외 지해하는 지난에서 안차에 따 | 0.0003	False
    습니다. 문 대통령은 또 평창 올림픽을 평화 올림픽으로 만들어, 북핵 | 습니다. 문 대통령은 또 평창 올림픽을 평화 올림픽으로 만들어, 북핵 | 0.7737	True
    지하철 7호선 방화사건을 수사중인 경기도 광명경찰서는 사건당일 | 지하철 7호선 방화사건을 수사중인 경기도 광명경찰서는 사건당일 | 0.9237	True
    은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 0.9452	True
    --------------------------------------------------------------------------------
    [52000/100000] Train loss: 0.00128, Valid loss: 1.09654, Elapsed_time: 20025.68035
    Current_accuracy : 83.100, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 꼬이기 시작했다"며 "문 후보도 그 부분에 대한 아쉬움을 얘기했고 최 | 0.8396	True
    원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 0.9535	True
    속할 수 있지만 본인이 사표를  이상 소환 등에 응할 가능성이 적어 | 속할 수 있지만 본인이 사표를  이상 소환 등에 응할 가능성이 적어 | 0.3117	True
    1년 만에 재개습니다. cj그룹 이미경 부회장 퇴진을 압박한 니 | 1년 만에 재개습니다. cj그룹 이미경 부회장 퇴진을 압박한 니 | 0.9250	True
    했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 0.8577	True
    --------------------------------------------------------------------------------
    [52500/100000] Train loss: 0.03844, Valid loss: 1.30110, Elapsed_time: 20217.78280
    Current_accuracy : 79.300, Current_norm_ED  : 0.91
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    근 노동현안에 대해 변화하는 만큼 믿고 지켜볼 것"이라고 말했다. 한 | 근 노동현안에 대해 변화하는 만큼 믿켜을 말했다. 양난 것다. 한 | 0.0001	False
    명숙 전 총리 사건처럼 정의가 거짓을 이긴다는 사실을 반드시 증명 | 일숙  전 의리 처사처럼 추거가 오짓면 남도가을 추련면 증정 | 0.0000	False
    뒤 취한 조니다. 법원에 청구한 동결 대상 재산은 28억 원에 매입한 | 뒤 취한 조니다. 법원에 청구한 동결 대상 재산은 28억 원에 매입한 | 0.6906	True
    하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 지고 있다고 밝다.인원 "월외희는 수관성지에서 경제희 보에 관계 | 0.0000	False
    로 정하고 추진했다. 이들 사업에 도는 사업당 200억 원을 한도로 20 | 나 정하고 추진했다. 이들 사업에 거르 사도에서도한 연로 원 | 0.0000	False
    --------------------------------------------------------------------------------
    [53000/100000] Train loss: 0.00406, Valid loss: 1.07301, Elapsed_time: 20408.06198
    Current_accuracy : 83.300, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.92
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 의도 받고 있다. 검찰은 박씨가 주가조작으로 거액의 시세차익을 | 0.9371	True
    과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 과 같은 논의구조가 아 노동계의 의견이 반영 수 있는 수준으로 | 0.9279	True
    "나흘 간의 설 연휴가 시작습니다. 연휴 첫 날, 잘 보내고 계신가요? | "나흘 간의 설 연휴가 시작습니다. 연휴 첫 날, 잘 보내고 계신가요? | 0.1704	True
    당 의원들이 "어차피 내려줄 예산인데 도지사 생색내기로 전락했다" | 당 의원들이 "어차피 내려줄 예산인데 도지사 생색내기로 전락했다" | 0.9141	True
    j그룹 회장에게 그 뜻을 전달했다고 진술했습니다. kbs 뉴스 이현기 | j그룹 회장에게 그 뜻을 전달했다고 진술했습니다. kbs 뉴스 이현기 | 0.9069	True
    --------------------------------------------------------------------------------
    [53500/100000] Train loss: 0.00272, Valid loss: 1.08052, Elapsed_time: 20600.62009
    Current_accuracy : 83.600, Current_norm_ED  : 0.93
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    로 정하고 추진했다. 이들 사업에 도는 사업당 200억 원을 한도로 20 | 로 정하다 고 하는 것으로 사행 사도에서는 그천당 20 | 0.0002	False
    0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 0일 열린 당정협의에서 황우여 원내대표는 "북한을 돕자, 북한 인권 | 0.8679	True
    오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 오후 10시30분께 "오늘은 소위를 열지않고 내일 오전 10시에 여야 | 0.7285	True
    어지기를 바란다"고 말했다. 한편, 신한카드는 이날 차병원에 소아뇌 | 념가  투계다. 이상돼 신소성 노동다. 이라, 대외 성 에 관계 와 | 0.0000	False
    성장위를 다루었으며, 명확한 임무가 부여되어 있다"고 말했다. 그는 | 성장위를 다루었으며, 명확한 임무가 부여되어 있다"고 말했다. 그는 | 0.9294	True
    --------------------------------------------------------------------------------
    [54000/100000] Train loss: 0.00242, Valid loss: 1.08182, Elapsed_time: 20791.59035
    Current_accuracy : 82.400, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 0.8840	True
    종 선정 등 이 두가지로 한정돼 있다"며 "(내가 경제수석시절에)동반 | 종 선정 등 이 두가지로 한정돼 있다"며 "(내가 경제수석시절에)동반 | 0.8806	True
    제협력국이 기획조정국, 총무국, 공보실과 함께 기획총 부문으로 | 제협력국이 기획조정국, 총무국, 공보실과 함께 기획총 부문으로 | 0.3220	True
    검찰 관계자는 "김 전 회장은 외교통상부와 재외 공관을 통해 인터 | 검찰 관계자는 "김 전 회장은 외교통상부와 재외 공관을 통해 인터 | 0.9446	True
    온 영하 6도까지 떨어지는 등 반짝 추위가 이어지습니다. 하지만, | 온 영하 6도까지 떨어지는 등 반짝 추위가 이어지습니다. 하지만, | 0.9426	True
    --------------------------------------------------------------------------------
    [54500/100000] Train loss: 0.00213, Valid loss: 1.07227, Elapsed_time: 20983.18415
    Current_accuracy : 83.200, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    7호 객차에 앉아 졸고 있었는데 갑자기 주위가 시끄러워 눈을 떠 보 | 7호 객차에 앉아 졸고 있었는데 갑자기 주위가 시끄러워 눈을 떠 보 | 0.0423	True
    습니다. 서해와 남해안에서는 짙은 안개를 주의하야습니다. 비 | 습니다. 서해에서 해검사 이해 여개적는도 했지해 한 사에서 생 | 0.0000	False
    28일 독립영화의 산실인 전주국제영화제가 화려한 막을 올린다. 올 | 28 금독립영화화 실인 전화영화국제영화화화 화합체화을 화린다. 화려을 올린 올 | 0.0000	False
    인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 0.9304	True
    나으며 한국노총과 민주노총 간부들도 회의장 안팎에서 진행상황 | 나으며 한국노총실 민주노총 간부들도 회의장 안팎지진 진행상황 | 0.0061	False
    --------------------------------------------------------------------------------
    [55000/100000] Train loss: 0.00182, Valid loss: 1.16106, Elapsed_time: 21174.27546
    Current_accuracy : 82.200, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 생인권법을 병합심사할 것을 주문하고 있다. 이 과정에서 한나라당은 | 0.8427	True
    니, 33년 만에 카타르에 지면서 8차전은 도하 참사로 기록습니다. | 니, 33년 만에 카타르에 지면 100년 도하는 사하로 했다. 검사 | 0.0000	False
    민생인권법을 북한인권법과 병합처리하자고 주장하는 것은 본질을 | 민생인권법을 북관하는 법이처리 합합하리하 집용본는 또은 본을을 | 0.0000	False
    제하는 것은 표현의 자유를 가로막을 뿐이며 정부가 반정부 시위 때 | 제하는 것은 표현의 자유를 가로막을 뿐이며 정부가 반정부 시위 때 | 0.9543	True
    의하야습니다. 아침 기온은 서울 영하 6도, 광주 영하 2도로 오늘 | 의하야습니다. 아침 기온은 서울 영하 6도, 광주 영하 2도로 오늘 | 0.9556	True
    --------------------------------------------------------------------------------
    [55500/100000] Train loss: 0.00146, Valid loss: 1.08840, Elapsed_time: 21365.40317
    Current_accuracy : 83.500, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    장경기록문화 테마파크 조성, 통영 국제음악당 건립, 김해 중소기업 | 장경기록문화 테마파크 조성, 통영 국제음악당 건립, 김해 중소기업 | 0.8090	True
    구조를 재개키로 했다. 한편 이날 좌초된 선박은 부산에서 석회석을 | 공류로 비 되시를 했다" 원편 은 전 공정부 대해팀 공정부  | 0.0000	False
    고 더 큰 사회 문제를 일으키는 원인이  것"이라며 "결국 그들이 당 | 고 더 큰 사회 문제를 일으키는 원인이  것"이라며 "결국 그들이 당 | 0.9445	True
    는 "노동계가 기대가 던 데 반해 참여정부가 한계가 있었다는 점을 | 는 "노동계가 기대가 던 데 반해 참여정부가 한계가 있었다는 점을 | 0.9263	True
    것"이라며 "이 두 가지 임무에 가시적인 진전이 있기를 희망한다"고 | 것"이라며 "이 두 가지 임무에 가시적인 진전이 있기를 희망한다"고 | 0.8999	True
    --------------------------------------------------------------------------------
    [56000/100000] Train loss: 0.00132, Valid loss: 1.05704, Elapsed_time: 21556.60237
    Current_accuracy : 82.800, Current_norm_ED  : 0.93
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    추가로 확보해 조사중이다. 경찰에 따르면 5일 밤 9시51분께 강원도 | 추가로 확보해 조사중이다. 경찰에 따르면 5일 밤 9시51분께 강원도 | 0.9729	True
    념이 충분히 반영돼 있다고 본다"고 주장했다. 또 "외통위를 통과해서 | 념이 충분히 반영돼 있다고 본다"고 주장했다. 또 "외통위를 통과해서 | 0.9496	True
    했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 했던 치즈축제와 오수의견제를 통합해 올해 처음 통합축제로 연다. | 0.8863	True
    9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 0.8734	True
    습니다. 바다의 물결은 동해 중부 먼바다에서 최고 4m까지 매우 높게 | 습니다. 바다의 물결은 동해 중부 먼바다에서 최고 4m까지 매우 높게 | 0.9266	True
    --------------------------------------------------------------------------------
    [56500/100000] Train loss: 0.00240, Valid loss: 1.11343, Elapsed_time: 21747.77986
    Current_accuracy : 83.200, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    오씨 학급만 공교롭게도 34명이어서 c군이 이 학급에 배정다고 해 | 오씨 학급만 공교롭게도 34명이어서 c군이 이 학급에 배정다고 해 | 0.8601	True
    이후 30년간 이 발전을 전세계에 알리는 계기가  것이라고 전망 | 이후 30년간 이 발전을 전세계에 알리는 계기가  것이라고 전망 | 0.8266	True
    에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 에서 "북인권법이라는 것을 기어코 조작해다면 그 순간부터 북남관 | 0.7449	True
    우유, 행복한 고객'을 실현하다고 선언했다. 75년을 넘어 100년으 | 우유, 행복한 고객'을 실현하다고 선언했다. 75년을 넘어 100년으 | 0.9176	True
    이 죽어다"며 "문 후보는 당시 민정수석으로 공권력 집행의 최종 결 | 이 죽어다"며 "문 후보는 당시 민정권석부는 공권력 집행의 최종 결 | 0.0009	False
    --------------------------------------------------------------------------------
    [57000/100000] Train loss: 0.00342, Valid loss: 1.25391, Elapsed_time: 21938.89988
    Current_accuracy : 78.700, Current_norm_ED  : 0.91
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    이 각각 생보검사국와 손보검사국으로 바뀌고 보험감독국이 분리된 | 이 각각 생보검사국와 손보검사국으로 바뀌고 보험감독국이 분리된 | 0.8113	True
    다. 한국정신대문제대책협의회 안정미 팀장은 "일본군 위안부 피해자 | 다. 한국정동대문제대책협의안 안정된 팀본부 민우층 피해자 할 해 | 0.0000	False
    만한 보양식이 없다. 칼칼한 낙지볶음에는 '크로포드 소비 블랑' | 만한 보양식이 없다. 칼칼한 낙지볶음에는 '크로포드 소비 블랑' | 0.8809	True
    . 내일은 오늘보다 기온이 더 높습니다. 서울 아침 기온 영하 6도, | . 내일은 오늘보다 기온이 더 높습니다. 서울 아침 기온 영하 6도, | 0.8770	True
    어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 | 어가 있기 때문에 이 법안을 그대로 의결하면 민주당이 제기한 북한 | 0.8547	True
    --------------------------------------------------------------------------------
    [57500/100000] Train loss: 0.00353, Valid loss: 1.12364, Elapsed_time: 22129.99568
    Current_accuracy : 83.300, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    충격패를 당하며 조 2위 자리가 위태로워지기도 했습니다.시리아와 | 충격패를 당하며 조 2위 자리가 위태로워지기도 했습니다.시리아와 | 0.8976	True
    다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 다. 금감원은 이밖에 금투자 권역에서도 감독 업무를 금투감독국으 | 0.8999	True
    념관 강당에서 발대식을 갖고, 4일부터 13일까지 9박 10일간의 여정 | 념관 강당에서 발대식을 갖고, 4일부터 13일까지 9박 10일간의 여정 | 0.2838	True
    날인 내일은 대부분 지방이 맑지만 제주도는 오후 한때 비나 눈이 | 날인 내일은 대부분 지방이 맑지만 제주도는 오후 한때 비나 눈이 | 0.8797	True
    오리엔테이션 자리에서 16학번 한 남학생이 신입생 후배에게 16학번 | 오리엔테이션 자리에서 16학번 한 남학생이 신입생 후배에게 16학번 | 0.9367	True
    --------------------------------------------------------------------------------
    [58000/100000] Train loss: 0.00229, Valid loss: 1.08439, Elapsed_time: 22321.92696
    Current_accuracy : 83.500, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    조 파업 때로 거슬러 올라 간다. 문 후보는 당시 청와대 민정수석으로 | 조 파업 때로 거슬러 올라 간다. 문 후보는 당시 청와대 민정수석으로 | 0.9143	True
    스럽고도 아 역사를 보고 듣고 느끼며 생생히 깨달을 수 있는 기회 | 스럽고도 아역 역사를 보고 듣고 느끼며 생생히 깨달을 수 있는 기회 | 0.0078	False
    블리인 리엄 페브르 그랑크 '레끌로'를 추천한다. 흰 과일, 꽃, 스 | 블리인 리엄 페브르 그랑크 '레끌로'를 추천한다. 흰 과일, 꽃, 스 | 0.7712	True
    제보다 5,6도 가량 낮습니다. 현재 대부분 해상으로 풍랑 특보가 발 | 제기다  6도 부량한 2씨도 부량 보도가 가비 특도가 연러 결도가 보 | 0.0000	False
    편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 0.8726	True
    --------------------------------------------------------------------------------
    [58500/100000] Train loss: 0.00133, Valid loss: 1.05431, Elapsed_time: 22512.07266
    Current_accuracy : 83.400, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    온 영하 6도까지 떨어지는 등 반짝 추위가 이어지습니다. 하지만, | 온 영하 6도까지 떨어지는 등 반짝 추위가 이어지습니다. 하지만, | 0.9561	True
    년께 독일에서 장협착증 수술을 받는등 건강이 악화돼 이듬해 1월께 | 년께 독일에서 장협착증 수술을 받는등 건강이 악화돼 이듬해 1월께 | 0.8878	True
    쟁력 강화 등 내수기반 강화가 필요하다"는 데 의견을 같이했다. 양 | 쟁력 강화 등 내수기반 강화가 필요하다"는 데 의견을 같이했다. 양 | 0.9333	True
    께 3개 국으로 늘어나게 된다. 저축은행 검사는 1, 2국으로 나다. | 께 3개 국으로 늘어나게 된다. 저축은행 검사는 1, 2국으로 나다. | 0.9224	True
    타깝다며 인성 교육을 더욱 강화하다고 밝다.학교 측은 이번 성 | 정깝면에 한나 것이 더욱 더위하기를 밝다.학교 측은 이 측은  | 0.0000	False
    --------------------------------------------------------------------------------
    [59000/100000] Train loss: 0.00135, Valid loss: 1.15685, Elapsed_time: 22702.19757
    Current_accuracy : 82.500, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    조금 오는 곳이 있습니다. 한낮 기온은 서울 4도, 대구는 8도 예상 | 조금 오는 곳이 있습니다. 한낮 기온은 서울 4도, 대구는 8도 예상 | 0.9746	True
    인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 0.9457	True
    한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 0.8541	True
    법을 "최악의 사태를 몰아오는 정치적 도발"로 규정했다. 북한은 논평 | 법을 "최악의 사태를 몰아오는 정치적 도발"로 규정했다. 북한은 논평 | 0.8250	True
    편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 편과 동반자살을 결심한 뒤 딸(11)에게 극약을 먹여 숨지게 해살인 | 0.8299	True
    --------------------------------------------------------------------------------
    [59500/100000] Train loss: 0.00126, Valid loss: 1.09730, Elapsed_time: 22895.19098
    Current_accuracy : 82.700, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 0.9415	True
    약인 '모자이크프로트사업'과 '인공습지조성사업' 관련 예산이 포 | 약인 '모자이크프로트사업'과 '인공습지조성사업' 관련 예산이 포 | 0.8810	True
    대피해 있다.                   | 대피해 있다.                   | 0.9876	True
    10조원을 합치면 30조원이 된다"고 밝다.  | 10조원을 합치면 30조원이 된다"고 밝다.  | 0.9305	True
    니다. 앞으로 대기는 점점 더 건조해 질 것으로 보여 화재가 나지 않도 | 니다. 앞으로 대기는 점점 더 건조해 질 것으로 보여 화재가 나지 않도 | 0.8139	True
    --------------------------------------------------------------------------------
    [60000/100000] Train loss: 0.00371, Valid loss: 1.17788, Elapsed_time: 23086.34841
    Current_accuracy : 81.900, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 러 다른 정관계 고위인사 등이 비리에 연루는지 추궁했다. | 0.8722	True
    조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 조씨와 현 의원이 돈의 규모와 성격을 '3억원이 아니라 활동비 명목 | 0.9394	True
    성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 0.8534	True
    하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 하고 있다고 밝다.인하대는 성희성폭력성차별 예방과 처리에 관 | 0.7235	True
    규모로 설립된 원자력 서비스센터는실물 원자로와 연료장전 설비, 가 | 규모로 설립된 원자력 서비스센터는실물 원자로와 연료장전 설비, 가 | 0.9076	True
    --------------------------------------------------------------------------------
    [60500/100000] Train loss: 0.00141, Valid loss: 1.12145, Elapsed_time: 23276.99620
    Current_accuracy : 82.600, Current_norm_ED  : 0.92
    Best_accuracy    : 84.000, Best_norm_ED     : 0.93
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    서 받은 자료를 바탕으로 책 두권 분량의 탄원서를 만들어 금감독 | 서 받은 자료를 바탕으로 책책바다. 또 서 의 토원들에 만질라 | 0.0000	False
    검찰 관계자는 "김 전 회장은 외교통상부와 재외 공관을 통해 인터 | 검찰 관계자는 "김 전 회장은 외교통상부와 재외 공관을 통해 인터 | 0.9625	True
    9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 9회 연속 월드컵 본선 진출은 마지막 벼랑 끝까지 몰리는 힘겨운 과정 | 0.9175	True
    인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 인하대 게시판에는 8일 '의대 남학우 9인의 성폭력을 고발합니다라 | 0.9463	True
    년께 독일에서 장협착증 수술을 받는등 건강이 악화돼 이듬해 1월께 | 년께 독일에서 장협착증 수술을을 받는등 건강이 악화돼 이듬해 1월께 | 0.1277	False
    --------------------------------------------------------------------------------
    ```
    

```go
!python3 train.py \
    --train_data ../data_aihub/lmdb/train \
    --valid_data ../data_aihub/lmdb/valid \
    --Transformation TPS \
    --FeatureExtraction VGG \
    --SequenceModeling BiLSTM \
    --Prediction Attn \
    --num_iter 100000 \
    --valInterval 500 \
    --batch_max_length 45 \
    --batch_size 128 \
    --data_filtering_off
```

## 학습 4 : TPS-ResNet-BiLSTM-Attn

- 학습결과분석
    - 5시간 40분 학습
    - params : 50800314
    - attention model이라 효과적인듯.
    - 아무래도 자연어 output이다보니, CTC 보다는 Attn 으로 학습을 하는게 효과적인듯.
    - 최고 정답률 92.2%
- 학습결과
    
    ```go
    --------------------------------------------------------------------------------
    dataset_root: ../data_aihub/lmdb/train
    opt.select_data: ['/']
    opt.batch_ratio: ['1']
    --------------------------------------------------------------------------------
    dataset_root:    ../data_aihub/lmdb/train	 dataset: /
    sub-directory:	/.	 num samples: 10000
    num total samples of /: 10000 x 1.0 (total_data_usage_ratio) = 10000
    num samples of / per batch: 128 x 1.0 (batch_ratio) = 128
    /usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py:563: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(
    --------------------------------------------------------------------------------
    Total_batch_size: 128 = 128
    --------------------------------------------------------------------------------
    dataset_root:    ../data_aihub/lmdb/valid	 dataset: /
    sub-directory:	/.	 num samples: 1000
    --------------------------------------------------------------------------------
    model input parameters 32 100 20 1 512 256 1010 45 TPS ResNet BiLSTM Attn
    Skip Transformation.LocalizationNetwork.localization_fc2.weight as it is already initialized
    Skip Transformation.LocalizationNetwork.localization_fc2.bias as it is already initialized
    Model:
    DataParallel(
      (module): Model(
        (Transformation): TPS_SpatialTransformerNetwork(
          (LocalizationNetwork): LocalizationNetwork(
            (conv): Sequential(
              (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
              (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (6): ReLU(inplace=True)
              (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (10): ReLU(inplace=True)
              (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (12): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (13): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (14): ReLU(inplace=True)
              (15): AdaptiveAvgPool2d(output_size=1)
            )
            (localization_fc1): Sequential(
              (0): Linear(in_features=512, out_features=256, bias=True)
              (1): ReLU(inplace=True)
            )
            (localization_fc2): Linear(in_features=256, out_features=40, bias=True)
          )
          (GridGenerator): GridGenerator()
        )
        (FeatureExtraction): ResNet_FeatureExtractor(
          (ConvNet): ResNet(
            (conv0_1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn0_1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv0_2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn0_2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (layer1): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (downsample): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
            )
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (layer2): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (downsample): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
            )
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (maxpool3): MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
            (layer3): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (downsample): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
              (2): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
              (3): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
              (4): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
            )
            (conv3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (layer4): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
              (1): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
              (2): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
              )
            )
            (conv4_1): Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), bias=False)
            (bn4_1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv4_2): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1), bias=False)
            (bn4_2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (AdaptiveAvgPool): AdaptiveAvgPool2d(output_size=(None, 1))
        (SequenceModeling): Sequential(
          (0): BidirectionalLSTM(
            (rnn): LSTM(512, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
          (1): BidirectionalLSTM(
            (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
            (linear): Linear(in_features=512, out_features=256, bias=True)
          )
        )
        (Prediction): Attention(
          (attention_cell): AttentionCell(
            (i2h): Linear(in_features=256, out_features=256, bias=False)
            (h2h): Linear(in_features=256, out_features=256, bias=True)
            (score): Linear(in_features=256, out_features=1, bias=False)
            (rnn): LSTMCell(1266, 256)
          )
          (generator): Linear(in_features=256, out_features=1010, bias=True)
        )
      )
    )
    Trainable params num :  50800314
    Optimizer:
    Adadelta (
    Parameter Group 0
        eps: 1e-08
        foreach: None
        lr: 1
        maximize: False
        rho: 0.95
        weight_decay: 0
    )
    ------------ Options -------------
    exp_name: TPS-ResNet-BiLSTM-Attn-Seed1111
    train_data: ../data_aihub/lmdb/train
    valid_data: ../data_aihub/lmdb/valid
    manualSeed: 1111
    workers: 4
    batch_size: 128
    num_iter: 100000
    valInterval: 1000
    saved_model: 
    FT: False
    adam: False
    lr: 1
    beta1: 0.9
    rho: 0.95
    eps: 1e-08
    grad_clip: 5
    baiduCTC: False
    select_data: ['/']
    batch_ratio: ['1']
    total_data_usage_ratio: 1.0
    batch_max_length: 45
    imgH: 32
    imgW: 100
    rgb: False
    character:  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률르른를름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없었엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄했행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘
    sensitive: False
    PAD: False
    data_filtering_off: True
    Transformation: TPS
    FeatureExtraction: ResNet
    SequenceModeling: BiLSTM
    Prediction: Attn
    num_fiducial: 20
    input_channel: 1
    output_channel: 512
    hidden_size: 256
    num_gpu: 1
    num_class: 1010
    ---------------------------------------
    
    [1/100000] Train loss: 6.92348, Valid loss: 6.94333, Elapsed_time: 4.70169
    Current_accuracy : 0.000, Current_norm_ED  : 0.00
    Best_accuracy    : 0.000, Best_norm_ED     : 0.00
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    양구경찰서에 육군 모 사단에서 복무중인 사병 이모(22)씨가 112전 | 름름름"름데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데데 | 0.0000	False
    오씨 학급만 공교롭게도 34명이어서 c군이 이 학급에 배정다고 해 | 력마엉엉엉엉엉엉엉엉엉엉엉엉엉도엉엉엉도엉엉도엉엉도엉도엉도엉도엉도엉도흐엉도엉도흐엉도엉 | 0.0000	False
    한 말맞추기라고 판단한 검찰은 즉각 조씨에 대한 사전 구속영장을 | 름름"""붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐붐 | 0.0000	False
    로 보여 개막식 행사에 참여하는 분들은 추위에 대비를 잘 해주야 | 력력"름"흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐흐 | 0.0000	False
    발견, 경찰에 신고 했다. 이씨는 경찰에서 "집에 돌아와 보니 함께 지 | 름뭄름빚빚빚빚빚뿌빚엉엉엉먼엉엉엉엉엉먼엉엉엉엉최엉엉엉엉엉최엉엉엉엉엉먼엉엉엉엉엉최엉엉 | 0.0000	False
    --------------------------------------------------------------------------------
    [1000/100000] Train loss: 4.56914, Valid loss: 5.15336, Elapsed_time: 823.59409
    Current_accuracy : 0.000, Current_norm_ED  : 0.20
    Best_accuracy    : 0.000, Best_norm_ED     : 0.20
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    부채 문제 등 취약요인에 대한 적극적인 대응이 필요한 시점이라는 | 로 있 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 | 0.0000	False
    의 정확한 판매량은 모르지만 1차분은 모두 판매된 것으로 알고 있 | 한 있 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 | 0.0000	False
    로 분리할 예정이다. 금감원은 조직개편과 함께 지원부서 인력을 현 | 로 있 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 전 | 0.0000	False
    이후 30년간 이 발전을 전세계에 알리는 계기가  것이라고 전망 | 다. 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 | 0.0000	False
    당하는 시공사 관계자도 붕괴위험을 사전에 지적하며 보강공사를 수 | 다. 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 이 | 0.0000	False
    --------------------------------------------------------------------------------
    [2000/100000] Train loss: 3.80090, Valid loss: 5.09878, Elapsed_time: 1647.03420
    Current_accuracy : 0.200, Current_norm_ED  : 0.29
    Best_accuracy    : 0.200, Best_norm_ED     : 0.29
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    술함을 여실히 드러다는 비판의 목소리가 높다. 5월 해평취수장의 | 을 를 불을 등 한 중을 중으로 '본 중동에 등 한 등 | 0.0000	False
    으나 현재 숨진 구씨를 제외한 나머지 선원 6명은 모두 안전한 곳에 | 부부 중 등 등 등 등 등 등 등 등 등 등 전 전 전 전 있 | 0.0000	False
    "3.1절인 오늘은 어제와 달리 맑은 하늘이 드러나 있지만, 바람이 상 | ""이 은 은 수 수 이 이 이 이 이 이 이 이 이 이 | 0.0000	False
    장을 향해 또다시 목소리를 높다. 최 장관은 30일 기자간담회에서 | 을 불을 등을 등 한 있다. 또다. 경장에서 경원에서 정 | 0.0000	False
    인성 교과목을 운영해 인성 및 인권 존중의 교육 환경을 조성해나가 | 한 인을 인을 중을 중한 인인 인인을 인한 인인 인인 인 | 0.0000	False
    --------------------------------------------------------------------------------
    [3000/100000] Train loss: 2.46775, Valid loss: 4.05595, Elapsed_time: 2464.34125
    Current_accuracy : 0.800, Current_norm_ED  : 0.48
    Best_accuracy    : 0.800, Best_norm_ED     : 0.48
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    노조 파업에 공권력 투입을 사실상 결정했고, 김 위원장은 철도노조 | 조조 업업에 투사을 사실을 사실했다. 위 위원은 노동노장 | 0.0000	False
    나 경기전망이 불투명 한데다 지원은 한시적이고 고용은 계속유지해 | 나 기기기자에 불용한 지용지 지용지 지용적 이용가 고속 계계 | 0.0000	False
    군산시에서는 5월 4일부터 8일까지 5월의 보리밭, 추억속으로 안내 | 군산시에서는 5일 1일 4일 5월 5리의 보리리, 보리 | 0.0000	False
    고 했다.                     | 입니다.                      | 0.0224	False
    학과정을 운영하고 있으며, 지난해 약 2000여 명이 신청했을 정도로 | 자치들를 제당하고 있으로 지난 20000년 이 대입 입입했 | 0.0000	False
    --------------------------------------------------------------------------------
    [4000/100000] Train loss: 1.06714, Valid loss: 1.42407, Elapsed_time: 3278.27425
    Current_accuracy : 18.200, Current_norm_ED  : 0.86
    Best_accuracy    : 18.200, Best_norm_ED     : 0.86
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 수리는 행위"면서 "이번 6월 6회에서 북한인권법이 처리 수 있 | 0.0000	False
    유동업조합으로 시작한 서울우유는 이후 국내 낙농산업의 신을 선 | 유동업조합으로 시작한 서울우유는 이후 국내 낙농산업의 신을 선 | 0.0000	True
    것"이라며 "이 두 가지 임무에 가시적인 진전이 있기를 희망한다"고 | 것"이라며 "이 두 가가 가무에 가시적인 전전이 있기를 "망한다"고 | 0.0000	False
    스 사진도 곁들여진다. '우리형'은 일본 uip를 통해 5월 28일 일본 | 스 사사도 진진진진다. '우리 일일은 일본 평결 8월 일본 88일 일본 | 0.0000	False
    두확인한 결과, 김우중 전 회장이 출국한 이후 어느 쪽 여권으로도 귀 | 두확인한 결과, 김우중 중이 회장한 출국 국국 어어 여 여권으로 도 | 0.0000	False
    --------------------------------------------------------------------------------
    [5000/100000] Train loss: 0.33875, Valid loss: 0.69056, Elapsed_time: 4095.25788
    Current_accuracy : 72.500, Current_norm_ED  : 0.94
    Best_accuracy    : 72.500, Best_norm_ED     : 0.94
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    의 일정 비율을 적립해 기부하게 된다. 차병원은 이기부금을 소아뇌 | 의 일정 비율을 적립해 기부하게 된다. 차병원은 이기부금을 소아뇌 | 0.0246	True
    결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 0.0236	True
    한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 한 뒤 690억여원의 시세차익을 챙기고 회돈 780억원을 령한  | 0.0014	True
    3건 25억7000여만원을 삭감했다. 여기에는 김 지사의 대표적인 공 | 3건 25억7000여만원을 삭감했다. 여기에는 김 지사의 대표적인 공 | 0.0088	True
    중소기업에 대해 근로시간 단축 지원금을 주고 있는데도 울산지역 | 중소기업에 대해 근로시간 단축 지원금을 주고 있는데도 울산지역 | 0.0023	True
    --------------------------------------------------------------------------------
    [6000/100000] Train loss: 0.14807, Valid loss: 2.17747, Elapsed_time: 4904.94043
    Current_accuracy : 5.000, Current_norm_ED  : 0.76
    Best_accuracy    : 72.500, Best_norm_ED     : 0.94
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    성마비 환자 의료 지원 및 불임 여성, 취약계층 의료비 지원 사업 등에 | 성마비 환자 의료 지원 및 불임 여성, 취약계층 층료비 의원 사업 등에 등에 등에 | 0.0000	False
    러나면서 현 전 의원이 종착지인지 여부를 밝히는 데 조씨의 진술이 | 러나면서 현 전 의원이 종착지인지 여부를 밝히는 조 조씨의 이술와 진 조씨이 이씨 | 0.0000	False
    의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은폐를 위 | 의 500만원'이라고 일치된 주장을 하는 것과 관련해 진술 은장을 위기 진폐를 위 | 0.0000	False
    몸과 마음이 지치기 쉬운 여름이다. 기력을 보충하기 위해 보양식을 | 몸과 마음이 지치기 쉬운 여름이다. 기력을 보충기 위해 보충하기 위해 보양을 | 0.0000	False
    이 초등학교 친구들보다 1년 늦었다. 아들이 동네 학교에서 친구들의 | 이 초등학학 친구들보다다. 1년 다. 아들이 아들 동교 학교 학구 등구 의구들의  | 0.0000	False
    --------------------------------------------------------------------------------
    [7000/100000] Train loss: 0.07863, Valid loss: 0.42208, Elapsed_time: 5717.68278
    Current_accuracy : 89.200, Current_norm_ED  : 0.97
    Best_accuracy    : 89.200, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    기술과 장비 개발에도 크게 도움이  것으로 기대된다. 두산중공업 | 기술과 장비 개발에도 크게 도움이  것으로 기대된다. 두산중공업 | 0.3847	True
    대중공업 등 대기업 협력업체들이 모기업에 따라 불가피하게 근로시 | 대중공업 등 대기업 협력업체들이 모기업에 따라 불가피하게 근로시 | 0.0814	True
    4년차 이상 실무자로 6개월 간 현지 문화와 언어를 익히고 시장 조사 | 4년차 이상 실무자로 6개월 간 현지 문화와 언어를 익히고 시장 조사 | 0.0593	True
    유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 0.3118	True
    규모로 설립된 원자력 서비스센터는실물 원자로와 연료장전 설비, 가 | 규모로 설립된 원자력 서비스센터는실물 원자로와 연료장전 설비, 가 | 0.0668	True
    --------------------------------------------------------------------------------
    [8000/100000] Train loss: 0.04548, Valid loss: 0.44898, Elapsed_time: 6530.07631
    Current_accuracy : 89.100, Current_norm_ED  : 0.97
    Best_accuracy    : 89.200, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    구조를 재개키로 했다. 한편 이날 좌초된 선박은 부산에서 석회석을 | 구조를 재려키로 했다. 한편 살부 언한 선본한 안안에서 동석을 | 0.0000	False
    원장, 감사원장에게 보 사실은 인정하고 있으며 이는 고문변호사로 | 원장, 감사원장에게 보 사실은 인정하고 있으며 이는 고문변호사로 | 0.7841	True
    다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 다"고 밝다. 현장에서는 '원 사진전'도 개최된다. 사진집에 실린 | 0.3301	True
    롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 0.1835	True
    스럽고도 아 역사를 보고 듣고 느끼며 생생히 깨달을 수 있는 기회 | 스럽고도 아 역사를 보고 듣고 느끼며 생생히 깨달을 수 있는 기회 | 0.1913	True
    --------------------------------------------------------------------------------
    [9000/100000] Train loss: 0.04051, Valid loss: 0.35623, Elapsed_time: 7339.66044
    Current_accuracy : 91.100, Current_norm_ED  : 0.97
    Best_accuracy    : 91.100, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    로 고용했으며 로르 회장은 2003년 이래 김 전회장을 최소 세 번 만 | 로 고용했으며 로르 회장은 2003년 이래 김 전회장을 최소 세 번 만 | 0.6428	True
    한국수자원공사의 물관리에 구멍이 뚫다. 두달새 같은 지점에서 유 | 한국수자원공사의 물관리에 구멍이 뚫다. 두달새 같은 지점에서 유 | 0.0971	True
    족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 록 내해에서서 강동구 승소도를 로도를 파류 것 따류 따류 시 | 0.0000	False
    건이 발생했다. 21일 현지 언론에 따르면 강도단은 이날 새벽 훔친 트 | 건이 발생했다. 21일 현지 언론에 따르 강도단은 이날 새날 훔친 트 | 0.0006	False
    감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 0.8049	True
    --------------------------------------------------------------------------------
    [10000/100000] Train loss: 0.06306, Valid loss: 0.36239, Elapsed_time: 8155.37723
    Current_accuracy : 91.200, Current_norm_ED  : 0.97
    Best_accuracy    : 91.200, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    귀국치 않다.                   | 귀국치 않다.                   | 0.9068	True
    정보습니다."                   | 정보습니다."                   | 0.9180	True
    이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 이 서울의 공개된 장소에서 김우중 전 대우그룹 회장을 만다는 외 | 0.6703	True
    그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 그룹은 최근 서울대와 손잡고 '밀크플러스'를 출시하며 기능성 우유 | 0.3418	True
    롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 롯데백화점이 '안중근 의사 해외 독립운동 유적지 방'을 후원한다. | 0.2881	True
    --------------------------------------------------------------------------------
    [11000/100000] Train loss: 0.01463, Valid loss: 0.40927, Elapsed_time: 8968.52579
    Current_accuracy : 90.100, Current_norm_ED  : 0.97
    Best_accuracy    : 91.200, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    글로스 주식 77억원 어치를 매수하는 과정에서 박 씨와 짜고 주가 | 글로스 주식 77억원 어치를 매수하는 과정에서 박 씨와 짜고 주가 | 0.3048	True
    상태로 이뤄지는 콜드체인시스템을 도입해 2000여 곳의 전용목장에 | 상태로 이뤄지는 콜드체인시스템을 도입해 2000여 곳의 전용목장에 | 0.6304	True
    사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 0.6001	True
    는데 회동 장소 중 한 곳이 서울이었다고 전했다. 1999년 10월 출 | 는데 회동 장소 중 한 곳이 서울이었다고 전했다. 1999년 10월 출 | 0.6261	True
    감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 감찰이 계속 가능성은사실상 희박한 것으로 판단하고 있다. 그러나 | 0.8556	True
    --------------------------------------------------------------------------------
    [12000/100000] Train loss: 0.03945, Valid loss: 0.40818, Elapsed_time: 9774.90894
    Current_accuracy : 91.400, Current_norm_ED  : 0.97
    Best_accuracy    : 91.400, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    다"고 벼르고 있다. 민주당은 "한나라당 법안은 보수단체 지원내용만 | 다"고 벼르고 있다. 민주당은 "한나라당 법안은 보수단체 지원내용만 | 0.7103	True
    사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 0.6008	True
    로 고용했으며 로르 회장은 2003년 이래 김 전회장을 최소 세 번 만 | 로 고용했으며 로르 회장은 2003년 이래 김 전회장을 최소 세 번 만 | 0.8124	True
    에 비기면서 출발부터 불안했습니다.4차전 이란 원정에서 단 한 개의 | 에 비기면서 출발부터 불안했습니다.4차전 이란 원정에서 단 한 개의 | 0.8505	True
    사 등 다른 사안과 연계 가능성까지 언급해 민주당의 반발을 고, 북 | 사 등 다른 사안과 연계 가능성까지 언급해 민주당의 반발을 고, 북 | 0.7020	True
    --------------------------------------------------------------------------------
    [13000/100000] Train loss: 0.01312, Valid loss: 0.68166, Elapsed_time: 10582.37357
    Current_accuracy : 82.700, Current_norm_ED  : 0.95
    Best_accuracy    : 91.400, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    받고 있다. 권 수석은 전화 통화를 인정했고 "일언지하에 거절했다"고 | 받고 있다. 권 수석은 전화 통화를 인정했고 "일언지하에 거절했다"고 | 0.7155	True
    오리엔테이션 자리에서 16학번 한 남학생이 신입생 후배에게 16학번 | 오리엔테이션 자리에서 16학번 한 남학생이 신입생 후배에게 16학번 | 0.3001	True
    하는 점에 비 b고교의 전입학 시스템에 중대 허점이 있었던 사실도 | 는는 점의 비은 b의 의학교의 아이 중원대에 대히 던실이 | 0.0000	False
    당히 강하게 불고 있습니다. 현재 동해안으로는 강풍 경보가, 그 밖에 | 당히 강하게 불고 있습니다. 현재 동해안으로는 강풍 경보가, 그 밖에 | 0.7954	True
    대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 대변인을 맡아 투쟁을 이끌었다. 김 위원장은 7일 내일신문과 통화에 | 0.5491	True
    --------------------------------------------------------------------------------
    [14000/100000] Train loss: 0.01456, Valid loss: 0.44787, Elapsed_time: 11400.88662
    Current_accuracy : 89.400, Current_norm_ED  : 0.97
    Best_accuracy    : 91.400, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    남북갈등으로까지 확산 조짐마저 보이고 있다.  | 남북갈등으로까지 확산 조짐마저 보이고 있다.  | 0.5323	True
    적 코리코 303호(860t급)의 선원 7명 가운데 이날 오후 7시 40분께 | 적 코리코 303호(860t급)의 선원 7명은 운운 오날 7시 7업 | 0.0001	False
    입니다. 다만 제주도는 일요일 오후부터 다시 비가 오기 시작하습 | 입니다. 다만 제주도는 일요일 오후부터 다시 비가 오기 시작하습 | 0.7757	True
    부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 부에서는 시큰둥했다"고 말했다. 또 다른 관계자도 "노무현정부에서 | 0.8330	True
    두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 두봉씨는 "행사 수익금 전액을 유니세프에 기부하기로 했다. 사진집 | 0.5358	True
    --------------------------------------------------------------------------------
    [15000/100000] Train loss: 0.04323, Valid loss: 0.34118, Elapsed_time: 12199.93790
    Current_accuracy : 91.900, Current_norm_ED  : 0.97
    Best_accuracy    : 91.900, Best_norm_ED     : 0.97
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 결정적이기 때문이다. 조씨는 검찰 조사에서 정씨를 만지만 돈은 | 0.7475	True
    오씨 학급만 공교롭게도 34명이어서 c군이 이 학급에 배정다고 해 | 오씨 학급만 공교롭게도 34명이어서 c군이 이 학급에 배정다고 해 | 0.7288	True
    이 가동돼 한은의 통화정책이 영향을 받는 것 아니냐는 우려의 목소 | 이 가동돼 한은의 통화정책이 영향을 받는 것 아니냐는 우려의 목소 | 0.8342	True
    편적 복지를 할 수 있는데 한 의 증세 없이도 마련할 수 있다"며 "부 | 편적 복지를 할 수 있는데 한 의 증세 없이도 마련할 수 있다"며 "부 | 0.7601	True
    사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 사사고가 발생해 구미시민들과 구미공단 입주업체들에게 큰 피해를 | 0.6298	True
    --------------------------------------------------------------------------------
    [16000/100000] Train loss: 0.00608, Valid loss: 0.32013, Elapsed_time: 13018.65102
    Current_accuracy : 91.900, Current_norm_ED  : 0.98
    Best_accuracy    : 91.900, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    어학교육 프로그램을 운영하고 있다. 희망자들을 대상으로 토익, 중 | 어학교고 프로 구을 운영하고 있다. 현망들분 대상 등석상으로 중익 | 0.0000	False
    고 했다.                     | 소했다.                      | 0.2458	False
    정권자는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김 | 정권자는데 이에 대한 사과가 우선"이라고 했다. 한편 문 후보와 김 | 0.9003	True
    검찰에 비공식적으로 자진귀국 의사를 타진한 적이있었으나 같은해 | 검찰에 비공식적으로 자진귀국 의사를 타진한 적이있었으나 같은해 | 0.8748	True
    부와 한은 간에 자료협조, 경제상황에 대한 의견교환 등 보다 긴밀한 | 부와 한은 간에 자료협조, 경제상황에 대한 의견교환 등 보다 긴밀한 | 0.8281	True
    --------------------------------------------------------------------------------
    [17000/100000] Train loss: 0.00826, Valid loss: 0.47750, Elapsed_time: 13837.06302
    Current_accuracy : 89.100, Current_norm_ED  : 0.97
    Best_accuracy    : 91.900, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 은 미세 먼지 농도가 일시적으로 높게 나타날 수 있기 때문에 따로 대 | 0.8217	True
    임원급 인사와 함께 발표할 예정이다. 개편안에 따르면 기존 감독서 | 원원급 인사에 함께 연정할 예표이다. 감찰안에 따르면 기 기독 표 | 0.0000	False
    회복을 중심으로 거시정책이 운영돼야 한다는 공통인식에 따라 정책 | 회복을 중심으로 거시정책이 운영돼야 한다는 공통인식에 따라 정책 | 0.7338	True
    충격패를 당하며 조 2위 자리가 위태로워지기도 했습니다.시리아와 | 충격패를 당하며 조 2위 자리가 위태로워지기도 했습니다.시리아와 | 0.2787	True
    물의를 빚어 감찰 대상에 올던 c검사가 이날 사표를 제출했다고 밝 | 물의를 빚어 감찰 대상에 올던 c검사가 이날 사표를 제출했다고 밝 | 0.8325	True
    --------------------------------------------------------------------------------
    [18000/100000] Train loss: 0.01341, Valid loss: 0.49427, Elapsed_time: 14649.50206
    Current_accuracy : 89.400, Current_norm_ED  : 0.97
    Best_accuracy    : 91.900, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    이미경 부회장 퇴진을 강요한 의를 받는 조원동 전 청와대 경제수 | 이미경 부회장 퇴진을 강요한 의를 받는 조원동 전 청와대 경제수 | 0.6074	True
    학과정을 운영하고 있으며, 지난해 약 2000여 명이 신청했을 정도로 | 학학정을 운영하고 있으며 지난해 지 처0여 여이라 여내해 정으로 | 0.0000	False
    파도가 높고 날이 어두워지자 이날 구조작업은 철수하고 17일 오전 | 파도가 가고 어이리과 어날 지날지로로 구조하업시 1수하고 있다. 두해 않속 | 0.0000	False
    김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 김두관 경남도지사가 '여소야대'로 곤욕을 치르고 있다. 최근 한나라 | 0.5142	True
    00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 0.6282	True
    --------------------------------------------------------------------------------
    [19000/100000] Train loss: 0.00640, Valid loss: 0.33573, Elapsed_time: 15458.82232
    Current_accuracy : 92.000, Current_norm_ED  : 0.98
    Best_accuracy    : 92.000, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    나으며 한국노총과 민주노총 간부들도 회의장 안팎에서 진행상황 | 나으며 한국노총과 민주노총 간부들도 회의장 안팎에서 진행상황 | 0.9069	True
    양구경찰서에 육군 모 사단에서 복무중인 사병 이모(22)씨가 112전 | 양구경찰서에 육군 모 사단에서 복무중인 사병 이모(22)씨가 112전 | 0.8006	True
    건강하고 품위있는 노년을 위한 정책을 만들다고 말했습니다. 어르 | 건강하고 품위있는 노년을 위한 정책을 만들다고 말했습니다. 어르 | 0.8837	True
    법무부 공보관은 "김우중씨 명의로 된 한국 여권과 프랑스 여권을 모 | 법무부 공보관은 "김우중씨 명의로 된 한국 여권과 프랑스 여권을 모 | 0.9164	True
    검찰에 비공식적으로 자진귀국 의사를 타진한 적이있었으나 같은해 | 검찰에 비공식적으로 자진귀국 의사를 타진한 적이있었으나 같은해 | 0.9051	True
    --------------------------------------------------------------------------------
    [20000/100000] Train loss: 0.00684, Valid loss: 0.35524, Elapsed_time: 16275.09250
    Current_accuracy : 92.000, Current_norm_ED  : 0.97
    Best_accuracy    : 92.000, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    감찰부는 c검사의 사표가 수리 때까지는 c검사에 대한 감찰을 계 | 감찰부는 c검사의 사표가 수리 때까지는 c검사에 대한 감찰을 계 | 0.9474	True
    유효슈팅 없이 0대 1로 완패한 데 이어, 6차전에서 다시 만난 중국에 | 유효슈팅 없이 0대 1로 완패한 데 이어, 6차전에서 다시 만난 중국에 | 0.8295	True
    님으로 와 알게된 정모씨에게 부탁해 c군의 주소를 명일동 소재 정씨 | 님으로 와 알게된 정모씨에게 부탁해 c군의 주소를 명일동 소재 정씨 | 0.8658	True
    로 가는 중심가치를 '행복'으로 설정한 것이다. 송용헌 서울우유협동 | 로 가는 중심가치를 '행복'으로 설정한 것이다. 송용헌 서울우유협동 | 0.8243	True
    는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 0.8794	True
    --------------------------------------------------------------------------------
    [21000/100000] Train loss: 0.03178, Valid loss: 0.40237, Elapsed_time: 17083.19126
    Current_accuracy : 91.100, Current_norm_ED  : 0.98
    Best_accuracy    : 92.000, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 흐리는 행위"라면서 "이번 6월 국회에서 북한인권법이 처리 수 있 | 0.7132	True
    된다. 고무에 카본블주1을 섞은 일반타이어는 날씨가 추워질수록 | 된다. 고무에 카본블주1을 섞은 일반타이어는 날씨가 추워질수록 | 0.6027	True
    있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 있어 매운 맛을 상쇄시켜 준다.  크로포드는 뉴질랜드 소비 블랑 | 0.7270	True
    원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 원을 직접 만나 돈을 건네지 않을 가능성이 높다고 보고 현 전 의원 | 0.8300	True
    00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 00만원을 받은 의(특경가법상 알선수재)로 은 전 위원에 대해 구속 | 0.8000	True
    --------------------------------------------------------------------------------
    [22000/100000] Train loss: 0.00532, Valid loss: 0.35675, Elapsed_time: 17891.28458
    Current_accuracy : 91.500, Current_norm_ED  : 0.98
    Best_accuracy    : 92.000, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 내세다. 남양, 매일, 그레 등 주식회사들도 빠른 의사결정을 무기 | 0.8676	True
    건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 건섭 금투서비스국장이 각각 승진 예정이다. 또 신한은행 감사로 | 0.9019	True
    당히 강하게 불고 있습니다. 현재 동해안으로는 강풍 경보가, 그 밖에 | 당히 강하게 불고 있습니다. 현재 동해안으로는 강풍 경보가, 그 밖에 | 0.9333	True
    과 다르게 경영실적보고서를 작성해 높은 등급을 받다. 이 덕분에 | 과 다르게 경영실적보고서를 작성해 높은 등급을 받다. 이 덕분에 | 0.9008	True
    과 오씨가 서로 아는 사이라는 사실도 뒤늦게 우연히 알게다"고 덧 | 과 오씨가 서로 아는 사이라는 사실도 뒤늦게 우연히 알게다"고 덧 | 0.8627	True
    --------------------------------------------------------------------------------
    [23000/100000] Train loss: 0.01229, Valid loss: 0.43166, Elapsed_time: 18701.12294
    Current_accuracy : 91.300, Current_norm_ED  : 0.97
    Best_accuracy    : 92.000, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    낮부터 기온이 오름세를 보이면서 추위가 금세 누그러지습니다. 그 | 낮부터 기온이 추름를 보름세 금위가 보습면 금위전 금그러지 | 0.0000	False
    는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 는 사실을 아는 사람은 많지 않다. 와인에는 칼 마그네 칼 나트 | 0.7326	True
    다. 조 전 수석은 박 전 대통령이 이 부회장 퇴진을 지시했고 손경식 c | 다. 조 전 수석은 박 전 대통령이 이 부회장 퇴진을 지시했고 손경식 c | 0.7539	True
    사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 사회적 합의에 기초해서 가야한다고 생각해 동반성장위원회를 만든 | 0.2792	True
    유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 유약하게 대화만 추구하지 않다며 이같이 말했다고 청와대가 밝 | 0.8386	True
    --------------------------------------------------------------------------------
    [24000/100000] Train loss: 0.00396, Valid loss: 0.38711, Elapsed_time: 19507.41468
    Current_accuracy : 92.200, Current_norm_ED  : 0.98
    Best_accuracy    : 92.200, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    니다. 토요일과 일요일에도 날씨로 인한 큰 불편은 없을 것으로 보 | 니다. 토요일과 일요일에도 날씨로 인한 큰 불편은 없을 것으로 보 | 0.9151	True
    해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 해해주는 세심함에 반했다"면서 "소울 메이트처럼 천생연분이라 느 | 0.9103	True
    발견, 경찰에 신고 했다. 이씨는 경찰에서 "집에 돌아와 보니 함께 지 | 발견, 경찰에 신고 했다. 이씨는 경찰에서 "집에 돌아와 보니 함께 지 | 0.9171	True
    주민의 인권을 실질적으로 증진하고 국제적 기준에 따라 인도적 지원 | 주민의 인권을 실질적으로 증진하고 국제적 기준에 따라 인도적 지원 | 0.9302	True
    기술과 장비 개발에도 크게 도움이  것으로 기대된다. 두산중공업 | 기술과 장비 개발에도 크게 도움이  것으로 기대된다. 두산중공업 | 0.9251	True
    --------------------------------------------------------------------------------
    [25000/100000] Train loss: 0.02206, Valid loss: 0.52052, Elapsed_time: 20313.96412
    Current_accuracy : 88.400, Current_norm_ED  : 0.97
    Best_accuracy    : 92.200, Best_norm_ED     : 0.98
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 족 전체가서울 강동구 명일동으로 주소를 옮긴 것으로 서류에 기재돼 | 0.8721	True
    가 훌하다.                    | 기 수다.                     | 0.1501	False
    법을 "최악의 사태를 몰아오는 정치적 도발"로 규정했다. 북한은 논평 | 법을 "최악의 사태를 몰아오는 정치적 도발"로 규정했다. 북한은 논평 | 0.5727	True
    그러지습니다. 바다의 물결은 동해상에서 최고 2.5미터까지 비교 | 그러지습니다. 바다의 물결은 동해상에서 최고 2.5미터까지 비교 | 0.8530	True
    한 감찰조사를 진행하고 있지만 이번 사표가 수리되면 감찰은 정지된 | 한 감찰조사를 진행하고 있지만 이번 사표가 수리되면 감찰은 정지된 | 0.7395	True
    --------------------------------------------------------------------------------
    ```
    

```
!python3 train.py \
    --train_data ../data_aihub/lmdb/train \
    --valid_data ../data_aihub/lmdb/valid \
    --Transformation TPS \
    --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM \
    --Prediction Attn \
    --num_iter 100000 \
    --valInterval 1000 \
    --batch_max_length 45 \
    --batch_size 128 \
    --data_filtering_off
```

[Final Project Report](https://www.notion.so/Final-Project-Report-dd9a6e3882494d9f84bc7bde311cd235)

testset에 있는거는 잘읽음. 내 데이터셋은 개좆망

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20167b04eba1464a9da95f0f099b6bf9e0/Untitled%204.png)

![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20167b04eba1464a9da95f0f099b6bf9e0/Untitled%205.png)

데이터의 크기 차이 및 필기체 차이도 존재.

testset : 67,042바이트 jpg

내 data : 3,542 바이트 png

→ 내 data를 training 100개, valid 10개로 finetuning 해봤지만 효과 없음.

→ 화질이랑 다른거 조절해서 한번 + 데이터 섞어서 한번 고
