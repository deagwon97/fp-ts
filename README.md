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
│	 └── 지역 : (55개)    

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

│         ├── trend_model

│         └──  cycle_model  

│

├── lunch model    

│ 	└──  flow_pop  = flow_trend + flow_cycle

│         ├── trend_model

│         └──  cycle_model

│

└── evening model     

  	└──  flow_pop  = flow_trend + flow_cycle

​          ├── trend_model

​          └──  cycle_model