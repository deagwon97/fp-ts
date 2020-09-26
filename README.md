# fp-ts
predict floating population in time series.

## [Linux Tree]

fp-ts/  
│  
├── data/  #.gitignore  
│   ├── original/  
│   └── preprocess/  
│         ├── train.npy  
│         ├── validation.npy  
│         └── test.npy  
│   
├── models/  
│   ├── lstm.py  
│   ├── gru.py  
│   └── xgboost.py  
│  
├── preprocess.ipynb  
├── train.ipynb  
├── result_validation.ipynb  
│  
├── requirements.txt  
├── .gitignore  
└── README.md 

  

## [data description]

### data

├── morning  
├── lunch  
└── evening  

### time data
time : mornig, lunch, evening -> 각기 다른 모델을 사용    

HDONG_CD : 동(지역)코드  
flow_pop : 유동인구    
​	├── flow_trend : 유동인구의 추세 변화  
​	└── flow_cycle : 유동인구의 진동(주간)  

card_use : 카드 사용량  
weekday : 요일  
holiday : 공휴일 (0,1)  
day_corona : 일별 신규  확진자  
ondo : 온도  
subdo : 습도  
rain_snow : 강수량  
day of years (?)   

### non time data

time :  mornig, lunch, evening -> 각기 다른 모델을 사용    

HDONG_CD : 동(지역)코드  
tot_pop : 지역 평균 유동인구  
age_80U : 80세 이상 인구 비율  
AREA : 지역 면적     



### data _split

![](https://github.com/deagwon97/image_src/blob/master/img/time_notime_data_split.png?raw=true)
├── train_data  
│ 	├── 시간 : 2019년 2월 ~ 5월, 2020년 2월 ~ -40일 까지  
│	└── 지역 : (55개)    
├── validation_data  
│ 	├── 시간 : - 40일 ~ -20일  
│ 	└── 지역 : (7개)  
└─ test_data  
		├── 시간 : - 20일 ~ -1일  
		└── 지역 : (7개)  

## [model]

![](https://github.com/deagwon97/image_src/blob/master/img/time_series_model.png?raw=true)

│  
├── morning model  
│ 	└──  flow_pop  = flow_trend + flow_cycle
│     	    ├── trend_model
│              └──  cycle_model  
│
├── lunch model    
│ 	└──  flow_pop  = flow_trend + flow_cycle
│      	     ├── trend_model
│      	     └──  cycle_model
│
└── evening model     
  	└──  flow_pop  = flow_trend + flow_cycle
​          	     ├── trend_model
​        	     └──  cycle_model

## 프로세스 정리

1. 모든 지역, 시간대의 정보를 담고 있는 table 데이터 생성

2. 27 가지 데이터 생성

    a. "요일" 데이터 삼각함수 변환

    b. "일년 중 날짜"데이터 삼각함수 변환

    c.  다음과 같은 열 생성

    

    'HDONG_CD' : 지역 코드  
    'time' : 시간대("아침", "점심","저녁")    
    'flow_pop' : 유동인구  
    'card_use' : 카드사용량  
    'holiday' : 공유일(1), 평일(0)  
    'day_corona' : 코로나 신규 확진자 수  
    'ondo' : 온도  
    subdo' : 습도  
    'rain_snow' : 강수량, 적설량    
    'dayofyear_sin' :  1년 중 날짜의 sin성분      
    'dayofyear_cos', : 1년 중 날짜의 cos성분    
    'weekday_sin' : 요일의 sin 성분  
    'weekday_cos': 요일의 cos 성분

    

    d.  "아침", "점심","저녁" 분리

    e. 훈련, 검정, 평가 데이터 분리

    f. 훈련 데이터로 Standard scaler 학습

    g. 훈련, 검정, 평가 데이터 스케일링

    h. 시계열 입력, 비 시계열 입력, 시계열 출력 3가지 종류의 데이터로 분리

    

3. trend 예측하기

4. cycle 예측하기

5.  결합하기

6. size 복원하기

7. 

