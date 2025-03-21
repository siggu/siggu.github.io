---
layout: single
title: "로지스틱 회귀 분류기 튜토리얼"
cattegories: machine-learning
tags: ["머신러닝", "로지스틱 회귀 분류기"]
---

# **로지스틱 회귀 분류기 튜토리얼**

캐글에 있는 [Prashant Banerjee](https://www.kaggle.com/prashant111)라는 분이 작성한 로지스틱 회귀 분류기 튜토리얼[Logistic Regression Classifier Tutorial](https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial)을 정리하려고 한다.

<br>

## **파이썬을 이용한 로지스틱 회귀 분류기 튜토리얼**

> 이 커널에서는 파이썬과 사이킷런을 사용해 로지스틱 회귀(**Logistic Regression**)를 구현한다. 호주에 내일 비가 올지 예측하는 이진 분류 모델을 로지스틱 회귀 분류기로 구축한다.

<br>

## **목차**

[1. 로지스틱 회귀에 대한 소개](#1-로지스틱-회귀에-대한-소개)

[2. 로지스틱 회귀의 이해](#2-로지스틱-회귀의-이해)

[3. 로지스틱 회귀의 가정](#3-로지스틱-회귀의-가정)

[4. 로지스틱 회귀의 유형](#4-로지스틱-회귀의-유형)

[5. 필요한 라이브러리 불러오기](#5필요한-라이브러리-불러오기)

[6. 데이터셋 불러오기](#6-데이터셋-불러오기)

[7. 탐색적 데이터 분석](#7-탐색적-데이터-분석)

[8. 특성 벡터와 대상 변수 선언](#8-특성-벡터와-대상-변수-선언)

[9. 데이터를 학습 및 테스트 세트로 분리](#9-데이터를-학습-및-테스트-세트로-분리)

[10. 특성 공학](#10-특성-공학)

[11. 특성 스케일링](#11-특성-스케일링)

[12. 모델 학습](#12-모델-학습)

[13. 결과 예측](#13-결과-예측)

[14. 정확도 점수 확인](#14-정확도-점수-확인)

[15. 오차 행렬](#15-오차-행렬)

[16. 분류 지표](#16-분류-지표)

[17. 임계값 조정](#17-임계값-조정)

[18. ROC - AUC](#18-roc-auc)

[19. k-Fold Cross Validation](#19-k-fold-교차-검증)

[20. GridSearch CV를 사용한 하이퍼파라미터 최적화](#20-그리드-탐색-cv를-사용한-하이퍼파라미터-최적화)

[21. 결과 및 결론](#21-결과-및-결론)

[22. 참고 문헌](#22-참고-문헌)

<br>

## **1. 로지스틱 회귀에 대한 소개**

[목차](#목차)

데이터 과학자들이 새로운 분류 문제를 마주하게 될 때, 가장 먼저 떠오르는 알고리즘 중 하나가 로지스틱 회귀(<strong>Logistic Regression</strong>)이다. 이는 이산적인 클래스로 관측치를 예측하기 위해 사용되는 지도학습 분류 알고리즘이다. 실제로는 관측치를 다른 카테고리로 분류하는 데 사용된다. 따라서 출력은 이산적인 성격을 가지게 된다. <strong>로지스틱 회귀</strong>는 <strong>Logit 회귀</strong>라고도 불리며, 가장 간단하고 직관적이며 다재다능한 분류 알고리즘 중 하나이다. 분류 문제를 해결하는 데 사용된다.

<br>

## **2. 로지스틱 회귀의 이해**

[목차](#목차)

통계학에서 <strong>로지스틱 회귀 모델</strong>은 주로 분류 목적으로 사용되는 널리 사용되는 통계 모델이다. 이는 일련의 관측치가 주어지면 로지스틱 회귀 알고리즘을 사용하여 이러한 관측치를 두 개 이상의 이산적인 클래스로 분류하는 데 도움이 된다. 따라서 대상 변수는 이산적인 성격을 가지게 된다.

로지스틱 회귀 알고리즘은 다음과 같이 작동한다 -

<br>

### **선형 방정식 구현**

로지스틱 회귀 알고리즘은 독립 변수 또는 설명 변수와 함께 선형 방정식을 구현하여 반응 값을 예측한다. 예를 들어, 우리는 공부한 시간과 시험 합격 확률의 예를 고려한다. 여기서, 공부한 시간은 설명 변수이고 x1로 나타내며, 시험 합격 확률은 반응 또는 대상 변수이고 z로 나타낸다.

만약 우리가 하나의 설명 변수(x1)와 하나의 반응 변수(z)를 가진다면, 선형 방정식은 다음과 같은 수식으로 나타낼 수 있다.

```python
z = β0 + β1x1
```

여기서, 계수 `β0`와 `β1`은 모델의 매개변수이다.

만약 여러 개의 설명 변수가 있다면, 위의 방정식은 다음과 같이 확장될 수 있다.

```python
z = β0 + β1x1+ β2x2 + ... + βnxn
```

여기서 `β0`, `β1`, `β2` 및 `βn`은 모델의 매개변수이다.

따라서, 예측된 반응 값은 위의 방정식에 의해 결정되며, `z`로 표시된다.

<br>

### **시그모이드 함수**

z라는 예측된 응답 값은, 0과 1 사이의 확률값으로 변환된다. 예측된 값에서 확률값을 얻기 위해서 시그모이드 함수를 사용한다. 이 시그모이드 함수는 모든 실수 값을 0과 1 사이의 확률 값으로 매핑한다.

머신러닝에서, 시그모이드 함수는 예측 값을 확률 값으로 매핑하는 데 사용된다. 시그모이드 함수는 S 모양의 곡선을 가지며, 이를 시그모이드 곡선이라고도 한다.

시그모이드 함수는 로지스틱 함수의 특수한 경우 중 하나이다. 이는 다음 수식으로 나타낼 수 있다.

그래프로는 다음과 같이 시그모이드 함수를 표현할 수 있다.

### 시그모이드 함수

![Sigmoid Function](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

<br>

### **결정 경계<font size="2">Decision boundary</font>**

시그모이드 함수는 0과 1 사이의 확률 값을 반환한다. 이 확률 값은 `"0"` 또는 `"1"`인 이진 클래스로 매핑된다. 이 확률 값을 이진 클래스로 매핑하려면 임계값을 선택해야 한다. 이 임계값을 결정 경계(Decision boundary)라고 한다. 이 임계값 이상에서는 확률 값을 클래스 1로 매핑하고, 이하에서는 클래스 0으로 매핑한다.

수학적으로는 다음과 같이 표현할 수 있다.

```
p ≥ 0.5 => class = 1

p < 0.5 => class = 0
```

일반적으로 결정 경계는 0.5로 설정된다. 따라서, 만약 확률 값이 0.8(>0.5)이면 해당 관측치를 1로 매핑한다.(0.5보다 작을 때는 0으로 매핑) 이는 아래 그래프에서 표현된다.

![Decision boundary in sigmoid function](https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_sigmoid_w_threshold.png)

<br>

### **예측하기**

이제 로지스틱 회귀에서 시그모이드 함수와 결정 경계에 대해 알았으므로, 우리는 이 지식을 활용해 예측 함수를 작성할 수 있다. 로지스틱 회귀의 예측 함수는 관측값이 긍정적인 클래스 1일 확률을 반환한다. 이는 P(class=1)로 나타내며, 확률이 1에 가까워질수록 해당 관측값이 클래스 1에 속한다는 모델에 대한 더 높은 신뢰도를 가진다. 그렇지 않으면 클래스 0에 속한다고 할 수 있다.

<br>

## **3. 로지스틱 회귀의 가정**

[목차](#목차)

로지스틱 회귀 모델은 몇 가지 주요 가정이 필요하다. 이는 다음과 같다.

1. 로지스틱 회귀 모델은 종속 변수가 이진, 다항 또는 순서형이어야한다.

2. 각 관측치는 서로 독립적이어야한다. 즉, 관측치는 반복 측정에서 나오면 안된다.

3. 로지스틱 회귀 알고리즘은 독립 변수 간 다중공선성이 적거나 없어야한다. 즉, 독립 변수는 서로 높은 상관 관계를 가지면 안된다.

4. 로지스틱 회귀 모델은 독립 변수 및 로그 오즈의 선형성을 가정한다.

5. 로지스틱 회귀 모델의 성공은 샘플 크기에 따라 달려 있다. 일반적으로 높은 정확도를 얻으려면 큰 샘플 크기가 필요하다.

<br>

## **4. 로지스틱 회귀의 유형**

[목차](#목차)

로지스틱 회귀 모델은 대상 변수 범주에 따라 세 가지 그룹으로 분류할 수 있다. 이는 다음과 같다.

1. 이항 로지스틱 회귀
   이항 로지스틱 회귀에서 대상 변수는 두 가지 가능한 범주를 가지고 있다. 대표적인 범주 예시로는 yes 혹은 no, good 혹은 bad, true 혹은 false, spam 혹은 no spam 그리고 pass 혹은 fail이 있다.
2. 다항 로지스틱 회귀
   다항 로지스틱 회귀에서 대상 변수는 특정 순서가 없는 세 개 이상의 범주를 가진다. 따라서 세 개 이상의 명목 범주가 있다. 이에 대한 예시로는 과일의 종류 - 사과, 망고, 오렌지, 바나나 등이 있다.

3. 순서형 로지스틱 회귀
   순서형 로지스틱 회귀에서 대상 변수는 세 개 이상의 순서형 범주를 가진다. 즉, 범주 간에 내재적인 순서가 있다. 예를 들어 학생 성적을 poor, average, good 그리고 excellent로 분류할 수 있다.

<br>

## **5. 필요한 라이브러리 불러오기**

[목차](#목차)

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/weather-dataset-rattle-package/weatherAUS.csv

```python
import warnings

warnings.filterwarnings('ignore')
```

<br>

## **6. 데이터셋 불러오기**

[목차](#목차)

```python
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
```

<br>

## **7. 탐색적 데이터 분석**

[목차](#목차)

데이터를 분석해보자.

```python

# view dimensions of dataset

df.shape
```

    (142193, 24)

<br>

142193개의 인스턴스와 24개의 변수가 있음을 확인할 수 있다.

<br>

```python
# preview the dataset

df.head()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>...</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>...</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>...</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>...</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>...</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>...</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>

```python
col_names = df.columns

col_names
```

    Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
           'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
           'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
           'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
           'Temp3pm', 'RainToday', 'RainTomorrow'],
          dtype='object')

### **RISK_MM 특성 지우기**

RISK_MM 특성 변수를 데이터 세트에서 삭제해야한다는 데이터 세트 설명이 있으므로 다음과 같이 삭제해야한다.

```python
df.drop(['RISK_MM'], axis=1, inplace=True)
```

```python
# view summary of dataset

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 142193 entries, 0 to 142192
    Data columns (total 23 columns):
    Date             142193 non-null object
    Location         142193 non-null object
    MinTemp          141556 non-null float64
    MaxTemp          141871 non-null float64
    Rainfall         140787 non-null float64
    Evaporation      81350 non-null float64
    Sunshine         74377 non-null float64
    WindGustDir      132863 non-null object
    WindGustSpeed    132923 non-null float64
    WindDir9am       132180 non-null object
    WindDir3pm       138415 non-null object
    WindSpeed9am     140845 non-null float64
    WindSpeed3pm     139563 non-null float64
    Humidity9am      140419 non-null float64
    Humidity3pm      138583 non-null float64
    Pressure9am      128179 non-null float64
    Pressure3pm      128212 non-null float64
    Cloud9am         88536 non-null float64
    Cloud3pm         85099 non-null float64
    Temp9am          141289 non-null float64
    Temp3pm          139467 non-null float64
    RainToday        140787 non-null object
    RainTomorrow     142193 non-null object
    dtypes: float64(16), object(7)
    memory usage: 25.0+ MB

### **변수 유형**

이 섹션에서는 데이터 세트를 범주형 변수와 수치형 변수로 분류한다. 데이터 세트에는 범주형 및 수치형 변수가 혼합되어 있다. 범주형 변수는 데이터 유형이 객체이다. 수치형 변수는 데이터 유형이 float64이다.

우선 범주형 변수를 찾아보자.

```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

    There are 7 categorical variables

    The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

```python
# view the categorical variables

df[categorical].head()
```

<br>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>WNW</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>WSW</td>
      <td>W</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>NE</td>
      <td>SE</td>
      <td>E</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>W</td>
      <td>ENE</td>
      <td>NW</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### 범주형 변수 요약

- 날짜 변수가 있다. `Date` 열로 표시된다.

- 6개의 범주형 변수가 있다. 이들은 `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` 및 `RainTomorrow` 이다.

- `RainToday`와 `RainTomorrow` 두 개의 이진 범주형 변수가 있다.

- `RainTomorrow`는 대상 변수다.

<br>

### **범주형 변수 문제 탐색**

먼저, 범주형 변수를 살펴보자.

<br>

### 범주형 변수에서 결측치 찾기

```python

# check missing values in categorical variables

df[categorical].isnull().sum()
```

    Date                0
    Location            0
    WindGustDir     10326
    WindDir9am      10566
    WindDir3pm       4228
    RainToday        3261
    RainTomorrow     3267
    dtype: int64

<br>

```python
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

    WindGustDir     10326
    WindDir9am      10566
    WindDir3pm       4228
    RainToday        3261
    RainTomorrow     3267
    dtype: int64

<br>

데이터 세트에서 누락된 값이 포함 된 범주형 변수는 WindGustDir, WindDir9am, WindDir3pm 및 RainToday이며 총 4개다.

<br>

### **범주형 변수의 빈도수**

범주형 변수의 빈도수를 확인해보자.

<br>

```python
# view frequency of categorical variables

for var in categorical:

    print(df[var].value_counts())
```

    2013-04-01    49
    2014-01-17    49
    2014-12-01    49
    2014-06-26    49
    2013-07-19    49
                  ..
    2007-12-13     1
    2007-11-04     1
    2007-11-23     1
    2008-01-12     1
    2007-11-16     1
    Name: Date, Length: 3436, dtype: int64
    Canberra            3436
    Sydney              3344
    Adelaide            3193
    Melbourne           3193
    Darwin              3193
    Perth               3193
    Brisbane            3193
    Hobart              3193
    GoldCoast           3040
    Albury              3040
    Cairns              3040
    Launceston          3040
    Ballarat            3040
    MountGambier        3040
    Bendigo             3040
    Albany              3040
    Townsville          3040
    MountGinini         3040
    AliceSprings        3040
    Wollongong          3040
    Newcastle           3039
    Penrith             3039
    Tuggeranong         3039
    PearceRAAF          3009
    Watsonia            3009
    WaggaWagga          3009
    NorfolkIsland       3009
    Moree               3009
    Dartmoor            3009
    Witchcliffe         3009
    BadgerysCreek       3009
    Portland            3009
    Cobar               3009
    Sale                3009
    Richmond            3009
    Williamtown         3009
    MelbourneAirport    3009
    Mildura             3009
    PerthAirport        3009
    Nuriootpa           3009
    CoffsHarbour        3009
    Woomera             3009
    SydneyAirport       3009
    Walpole             3006
    NorahHead           3004
    SalmonGums          3001
    Nhil                1578
    Uluru               1578
    Katherine           1578
    Name: Location, dtype: int64
    W      9915
    SE     9418
    N      9313
    SSE    9216
    E      9181
    S      9168
    WSW    9069
    SW     8967
    SSW    8736
    WNW    8252
    NW     8122
    ENE    8104
    ESE    7372
    NE     7133
    NNW    6620
    NNE    6548
    Name: WindGustDir, dtype: int64
    N      11758
    SE      9287
    E       9176
    SSE     9112
    NW      8749
    S       8659
    W       8459
    SW      8423
    NNE     8129
    NNW     7980
    ENE     7836
    NE      7671
    ESE     7630
    SSW     7587
    WNW     7414
    WSW     7024
    Name: WindDir9am, dtype: int64
    SE     10838
    W      10110
    S       9926
    WSW     9518
    SSE     9399
    SW      9354
    N       8890
    WNW     8874
    NW      8610
    ESE     8505
    E       8472
    NE      8263
    SSW     8156
    NNW     7870
    ENE     7857
    NNE     6590
    Name: WindDir3pm, dtype: int64
    No     110319
    Yes     31880
    Name: RainToday, dtype: int64
    No     110316
    Yes     31877
    Name: RainTomorrow, dtype: int64

```python
# view frequency distribution of categorical variables

for var in categorical:

    print(df[var].value_counts()/np.float(len(df)))
```

    2013-04-01    0.000337
    2014-01-17    0.000337
    2014-12-01    0.000337
    2014-06-26    0.000337
    2013-07-19    0.000337
                    ...
    2007-12-13    0.000007
    2007-11-04    0.000007
    2007-11-23    0.000007
    2008-01-12    0.000007
    2007-11-16    0.000007
    Name: Date, Length: 3436, dtype: float64
    Canberra            0.023622
    Sydney              0.022989
    Adelaide            0.021951
    Melbourne           0.021951
    Darwin              0.021951
    Perth               0.021951
    Brisbane            0.021951
    Hobart              0.021951
    GoldCoast           0.020899
    Albury              0.020899
    Cairns              0.020899
    Launceston          0.020899
    Ballarat            0.020899
    MountGambier        0.020899
    Bendigo             0.020899
    Albany              0.020899
    Townsville          0.020899
    MountGinini         0.020899
    AliceSprings        0.020899
    Wollongong          0.020899
    Newcastle           0.020892
    Penrith             0.020892
    Tuggeranong         0.020892
    PearceRAAF          0.020686
    Watsonia            0.020686
    WaggaWagga          0.020686
    NorfolkIsland       0.020686
    Moree               0.020686
    Dartmoor            0.020686
    Witchcliffe         0.020686
    BadgerysCreek       0.020686
    Portland            0.020686
    Cobar               0.020686
    Sale                0.020686
    Richmond            0.020686
    Williamtown         0.020686
    MelbourneAirport    0.020686
    Mildura             0.020686
    PerthAirport        0.020686
    Nuriootpa           0.020686
    CoffsHarbour        0.020686
    Woomera             0.020686
    SydneyAirport       0.020686
    Walpole             0.020665
    NorahHead           0.020652
    SalmonGums          0.020631
    Nhil                0.010848
    Uluru               0.010848
    Katherine           0.010848
    Name: Location, dtype: float64
    W      0.068163
    SE     0.064746
    N      0.064024
    SSE    0.063358
    E      0.063117
    S      0.063028
    WSW    0.062347
    SW     0.061646
    SSW    0.060058
    WNW    0.056730
    NW     0.055837
    ENE    0.055713
    ESE    0.050681
    NE     0.049038
    NNW    0.045511
    NNE    0.045016
    Name: WindGustDir, dtype: float64
    N      0.080833
    SE     0.063846
    E      0.063083
    SSE    0.062643
    NW     0.060147
    S      0.059528
    W      0.058153
    SW     0.057906
    NNE    0.055885
    NNW    0.054860
    ENE    0.053870
    NE     0.052736
    ESE    0.052454
    SSW    0.052159
    WNW    0.050969
    WSW    0.048288
    Name: WindDir9am, dtype: float64
    SE     0.074508
    W      0.069504
    S      0.068239
    WSW    0.065434
    SSE    0.064616
    SW     0.064306
    N      0.061116
    WNW    0.061006
    NW     0.059192
    ESE    0.058470
    E      0.058243
    NE     0.056806
    SSW    0.056070
    NNW    0.054104
    ENE    0.054015
    NNE    0.045305
    Name: WindDir3pm, dtype: float64
    No     0.758415
    Yes    0.219167
    Name: RainToday, dtype: float64
    No     0.758394
    Yes    0.219146
    Name: RainTomorrow, dtype: float64

<br>

### **레이블의 수**

카테고리 변수 내 레이블의 수는 기수(`cardinality`)라고한다. 변수 내 레이블의 수가 많으면 높은 기수(`high cardinality`)라고한다. 높은 기수는 머신 러닝 모델에서 일부 심각한 문제를 일으킬 수 있으므로, 높은 기수를 확인해보자.

<br>

```python
# check for cardinality in categorical variables

for var in categorical:

    print(var, ' contains ', len(df[var].unique()), ' labels')
```

    Date  contains  3436  labels
    Location  contains  49  labels
    WindGustDir  contains  17  labels
    WindDir9am  contains  17  labels
    WindDir3pm  contains  17  labels
    RainToday  contains  3  labels
    RainTomorrow  contains  3  labels

<br>

다음과 같이 전처리가 필요해 보이는 `Date` 변수가 있다.

다른 모든 변수는 비교적 적은 수의 변수를 가지고 있다.

<br>

### **Data 변수의 특성 엔지니어링**

```python
df['Date'].dtypes
```

    dtype('O')

<br>

`Date` 변수의 데이터 타입이 `object`이므로, 이를 datetime 형식으로 파싱해보자.

```python
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
```

```python
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
```

    0    2008
    1    2008
    2    2008
    3    2008
    4    2008
    Name: Year, dtype: int64

<br>

```python
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
```

    0    12
    1    12
    2    12
    3    12
    4    12
    Name: Month, dtype: int64

<br>

```python
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
```

    0    1
    1    2
    2    3
    3    4
    4    5
    Name: Day, dtype: int64

<br>

```python
# again view the summary of dataset

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 145460 entries, 0 to 145459
    Data columns (total 26 columns):
    Date             145460 non-null datetime64[ns]
    Location         145460 non-null object
    MinTemp          143975 non-null float64
    MaxTemp          144199 non-null float64
    Rainfall         142199 non-null float64
    Evaporation      82670 non-null float64
    Sunshine         75625 non-null float64
    WindGustDir      135134 non-null object
    WindGustSpeed    135197 non-null float64
    WindDir9am       134894 non-null object
    WindDir3pm       141232 non-null object
    WindSpeed9am     143693 non-null float64
    WindSpeed3pm     142398 non-null float64
    Humidity9am      142806 non-null float64
    Humidity3pm      140953 non-null float64
    Pressure9am      130395 non-null float64
    Pressure3pm      130432 non-null float64
    Cloud9am         89572 non-null float64
    Cloud3pm         86102 non-null float64
    Temp9am          143693 non-null float64
    Temp3pm          141851 non-null float64
    RainToday        142199 non-null object
    RainTomorrow     142193 non-null object
    Year             145460 non-null int64
    Month            145460 non-null int64
    Day              145460 non-null int64
    dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
    memory usage: 28.9+ MB

<br>

`Data` 변수에서 세 개의 추가 열이 만들어진 것을 확인할 수 있습니다. 이제 데이터 세트에서 원래의 날짜 변수를 삭제하겠습니다.

<br>

```python
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
```

```python
# preview the dataset again

df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>...</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>...</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>...</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>...</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>...</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>

<br>

데이터셋에서 `Date` 변수가 삭제된 것을 볼 수 있다.

<br>

### **범주형 변수 탐색**

이제 하나씩 범주형 변수를 하나하나 탐색해보자.

```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

    There are 6 categorical variables

    The categorical variables are : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

<br>

6개의 범주형 변수들이 있는 것을 볼 수 있다. `Data` 변수는 사라졌다. 제일 먼저, 범주형 변수들의 결측치를 살펴보자.

```python
# check for missing values in categorical variables

df[categorical].isnull().sum()
```

    Location            0
    WindGustDir      9330
    WindDir9am      10013
    WindDir3pm       3778
    RainToday        1406
    RainTomorrow        0
    dtype: int64

<br>

`WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`, 이 4개가 결측치를 포함하고 있는 것을 볼 수 있다. 모든 변수들을 하나하나 살펴보자.

### `Location` 변수 탐색

```python
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
```

    Location contains 49 labels

<br>

```python
# check labels in location variable

df.Location.unique()
```

    array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
           'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
           'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
           'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
           'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
           'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
           'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
           'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
           'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
           'AliceSprings', 'Darwin', 'Katherine', 'Uluru'], dtype=object)

<br>

```python
# check frequency distribution of values in Location variable

df.Location.value_counts()
```

    Canberra            3418
    Sydney              3337
    Perth               3193
    Darwin              3192
    Hobart              3188
    Brisbane            3161
    Adelaide            3090
    Bendigo             3034
    Townsville          3033
    AliceSprings        3031
    MountGambier        3030
    Ballarat            3028
    Launceston          3028
    Albany              3016
    Albury              3011
    PerthAirport        3009
    MelbourneAirport    3009
    Mildura             3007
    SydneyAirport       3005
    Nuriootpa           3002
    Sale                3000
    Watsonia            2999
    Tuggeranong         2998
    Portland            2996
    Woomera             2990
    Cobar               2988
    Cairns              2988
    Wollongong          2983
    GoldCoast           2980
    WaggaWagga          2976
    NorfolkIsland       2964
    Penrith             2964
    SalmonGums          2955
    Newcastle           2955
    CoffsHarbour        2953
    Witchcliffe         2952
    Richmond            2951
    Dartmoor            2943
    NorahHead           2929
    BadgerysCreek       2928
    MountGinini         2907
    Moree               2854
    Walpole             2819
    PearceRAAF          2762
    Williamtown         2553
    Melbourne           2435
    Nhil                1569
    Katherine           1559
    Uluru               1521
    Name: Location, dtype: int64

<br>

```python
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Albany</th>
      <th>Albury</th>
      <th>AliceSprings</th>
      <th>BadgerysCreek</th>
      <th>Ballarat</th>
      <th>Bendigo</th>
      <th>Brisbane</th>
      <th>Cairns</th>
      <th>Canberra</th>
      <th>Cobar</th>
      <th>...</th>
      <th>Townsville</th>
      <th>Tuggeranong</th>
      <th>Uluru</th>
      <th>WaggaWagga</th>
      <th>Walpole</th>
      <th>Watsonia</th>
      <th>Williamtown</th>
      <th>Witchcliffe</th>
      <th>Wollongong</th>
      <th>Woomera</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>

<br>

### `WindGustDir` 변수 탐색

```python
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```

    WindGustDir contains 17 labels

<br>

```python
# check labels in WindGustDir variable

df['WindGustDir'].unique()
```

    array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
           'S', 'NW', 'SE', 'ESE', nan, 'E', 'SSW'], dtype=object)

<br>

```python
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
```

    W      9780
    SE     9309
    E      9071
    N      9033
    SSE    8993
    S      8949
    WSW    8901
    SW     8797
    SSW    8610
    WNW    8066
    NW     8003
    ENE    7992
    ESE    7305
    NE     7060
    NNW    6561
    NNE    6433
    Name: WindGustDir, dtype: int64

<br>

```python
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

    ENE    7992
    ESE    7305
    N      9033
    NE     7060
    NNE    6433
    NNW    6561
    NW     8003
    S      8949
    SE     9309
    SSE    8993
    SSW    8610
    SW     8797
    W      9780
    WNW    8066
    WSW    8901
    NaN    9330
    dtype: int64

<br>

`WindGustDir` 변수에 9330개의 결측치가 있는 것을 볼 수 있다.

### `WindDir9am` 변수 탐색

```python
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```

    WindDir9am contains 17 labels

<br>

```python
# check labels in WindDir9am variable

df['WindDir9am'].unique()
```

    array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
           'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)

<br>

```python
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
```

    N      11393
    SE      9162
    E       9024
    SSE     8966
    NW      8552
    S       8493
    W       8260
    SW      8237
    NNE     7948
    NNW     7840
    ENE     7735
    ESE     7558
    NE      7527
    SSW     7448
    WNW     7194
    WSW     6843
    Name: WindDir9am, dtype: int64

<br>

```python
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

    ENE     7735
    ESE     7558
    N      11393
    NE      7527
    NNE     7948
    NNW     7840
    NW      8552
    S       8493
    SE      9162
    SSE     8966
    SSW     7448
    SW      8237
    W       8260
    WNW     7194
    WSW     6843
    NaN    10013
    dtype: int64

<br>

`WindDir9am` 변수에 10013개의 결측치가 있는 것을 볼 수 있다.

<br>

### `WinDir3pm` 변수 탐색

```python
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```

    WindDir3pm contains 17 labels

<br>

```python
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
```

    array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
           'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)

<br>

```python
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
```

    SE     10663
    W       9911
    S       9598
    WSW     9329
    SW      9182
    SSE     9142
    N       8667
    WNW     8656
    NW      8468
    ESE     8382
    E       8342
    NE      8164
    SSW     8010
    NNW     7733
    ENE     7724
    NNE     6444
    Name: WindDir3pm, dtype: int64

<br>

```python
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

    ENE     7724
    ESE     8382
    N       8667
    NE      8164
    NNE     6444
    NNW     7733
    NW      8468
    S       9598
    SE     10663
    SSE     9142
    SSW     8010
    SW      9182
    W       9911
    WNW     8656
    WSW     9329
    NaN     3778
    dtype: int64

<br>

`WindDir3pm` 변수에는 3778개의 결측치가 있는 것을 볼 수 있다.

<br>

### `RainToday` 변수 탐색

```python
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```

    RainToday contains 3 labels

<br>

```python
# check labels in WindGustDir variable

df['RainToday'].unique()
```

    array(['No', 'Yes', nan], dtype=object)

<br>

```python
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
```

    No     109332
    Yes     31455
    Name: RainToday, dtype: int64

<br>

```python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

    Yes    31455
    NaN     1406
    dtype: int64

<br>

`RainToday` 변수에는 1406개의 결측치가 있는 것을 볼 수 있다.

<br>

### 수치형 변수 탐색

```python
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```

    There are 19 numerical variables

    The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']

<br>

```python
# view the numerical variables

df[numerical].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### 수치형 변수들 요약

- 16개의 수치형 변수들이 있다.

- 이들은 `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am`, `Temp3pm` 이다.
- 모든 수치형 변수들은 연속형 변수이다.

<br>

### **수치형 변수의 문제 탐색**

수치형 변수들을 탐색할 것이다.

<br>

### 수치형 특성의 결측치

```python
# check missing values in numerical variables

df[numerical].isnull().sum()
```

    MinTemp            637
    MaxTemp            322
    Rainfall          1406
    Evaporation      60843
    Sunshine         67816
    WindGustSpeed     9270
    WindSpeed9am      1348
    WindSpeed3pm      2630
    Humidity9am       1774
    Humidity3pm       3610
    Pressure9am      14014
    Pressure3pm      13981
    Cloud9am         53657
    Cloud3pm         57094
    Temp9am            904
    Temp3pm           2726
    Year                 0
    Month                0
    Day                  0
    dtype: int64

결측치를 포함한 16개의 수치형 변수들을 볼 수 있다.

### 수치형 변수의 이상치

```python
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
```

            MinTemp   MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
    count  141556.0  141871.0  140787.0      81350.0   74377.0       132923.0
    mean       12.0      23.0       2.0          5.0       8.0           40.0
    std         6.0       7.0       8.0          4.0       4.0           14.0
    min        -8.0      -5.0       0.0          0.0       0.0            6.0
    25%         8.0      18.0       0.0          3.0       5.0           31.0
    50%        12.0      23.0       0.0          5.0       8.0           39.0
    75%        17.0      28.0       1.0          7.0      11.0           48.0
    max        34.0      48.0     371.0        145.0      14.0          135.0

           WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
    count      140845.0      139563.0     140419.0     138583.0     128179.0
    mean           14.0          19.0         69.0         51.0       1018.0
    std             9.0           9.0         19.0         21.0          7.0
    min             0.0           0.0          0.0          0.0        980.0
    25%             7.0          13.0         57.0         37.0       1013.0
    50%            13.0          19.0         70.0         52.0       1018.0
    75%            19.0          24.0         83.0         66.0       1022.0
    max           130.0          87.0        100.0        100.0       1041.0

           Pressure3pm  Cloud9am  Cloud3pm   Temp9am   Temp3pm      Year  \
    count     128212.0   88536.0   85099.0  141289.0  139467.0  142193.0
    mean        1015.0       4.0       5.0      17.0      22.0    2013.0
    std            7.0       3.0       3.0       6.0       7.0       3.0
    min          977.0       0.0       0.0      -7.0      -5.0    2007.0
    25%         1010.0       1.0       2.0      12.0      17.0    2011.0
    50%         1015.0       5.0       5.0      17.0      21.0    2013.0
    75%         1020.0       7.0       7.0      22.0      26.0    2015.0
    max         1040.0       9.0       9.0      40.0      47.0    2017.0

              Month       Day
    count  142193.0  142193.0
    mean        6.0      16.0
    std         3.0       9.0
    min         1.0       1.0
    25%         3.0       8.0
    50%         6.0      16.0
    75%         9.0      23.0
    max        12.0      31.0   2

<br>

`Rainfall`, `Evaporation`, `WindSpeed9am`, `WindSpeed3pm`이 이상치를 가지고 있는 것으로 보인다.

위 변수들의 이상치를 시각화 해보자.

```python
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

    Text(0, 0.5, 'WindSpeed3pm')

![image](https://user-images.githubusercontent.com/106001755/231873176-79486bf8-2a57-40a2-849c-8bbf7f21abe1.png)

<br>

이 변수들 안에 많은 이상치가 있는 것을 확인할 수 있다.

<br>

### **변수의 분산 확인**

이제 히스토그램을 그려 분포를 확인하여 변수들이 정규분포를 따르는지, 혹은 치우친 분포를 보이는지를 확인해보자. 만약 변수가 정규분포를 따른다면, 극단치 분석(`Extreme Value Analysis`)을 수행할 것이고, 변수가 치우친 분포를 보인다면 IQR (Interquantile range)을 수행할 것이다.

<br>

```python
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

    Text(0, 0.5, 'RainTomorrow')

![image](https://user-images.githubusercontent.com/106001755/231873288-b47d7036-8532-44b8-8b4b-436f30d4e527.png)

<br>

4개 모두 치우친 것을 볼 수 있다. 따라서, IQR을 수행할 것이다.

<br>

```python
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    Rainfall outliers are values < -2.4000000000000004 or > 3.2

`Rainfall`에 대해, 최소값과 최대값이 0.0과 371.0이다. 따라서 이상치는 3.2보다 큰 값들이다.

<br>

```python
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    Evaporation outliers are values < -11.800000000000002 or > 21.800000000000004

`Evaporation`에 대해, 최소값과 최대값이 0.0과 145.0이다. 따라서 이상치는 21.8보다 큰 값들이다.

<br>

```python
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    WindSpeed9am outliers are values < -29.0 or > 55.0

`WindSpeed9am`에 대해, 최소값과 최대값이 0.0과 130.0이다. 따라서 이상치는 55.0보다 큰 값들이다.

<br>

```python
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

```

    WindSpeed3pm outliers are values < -20.0 or > 57.0

`WindSpeed3pm`에 대해, 최소값과 최대값이 0.0과 87.0이다. 따라서 이상치는 570보다 큰 값들이다.

<br>
<br>

## **8. 특성 벡터와 대상 변수 선언**

[목차](#목차)

```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

<br>

## **9. 데이터를 학습 및 테스트 세트로 분리**

[목차](#목차)

```python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

```

```python
# check the shape of X_train and X_test

X_train.shape, X_test.shape
```

    ((113754, 24), (28439, 24))

<br>
<br>

## **10. 특성 공학**

[목차](#목차)

**특성 공학**은 원시 데이터를 유용한 기능으로 변환하여 모델을 이해하고 예측 능력을 향상시키는 프로세스이다. 다른 유형의 변수에 대해 특성 공학을 수행해보자.

먼저, 범주형 변수와 수치형 변수를 다시 나눠서 보여줄 것이다.

```python
# check data types in X_train

X_train.dtypes
```

    Location          object
    MinTemp          float64
    MaxTemp          float64
    Rainfall         float64
    Evaporation      float64
    Sunshine         float64
    WindGustDir       object
    WindGustSpeed    float64
    WindDir9am        object
    WindDir3pm        object
    WindSpeed9am     float64
    WindSpeed3pm     float64
    Humidity9am      float64
    Humidity3pm      float64
    Pressure9am      float64
    Pressure3pm      float64
    Cloud9am         float64
    Cloud3pm         float64
    Temp9am          float64
    Temp3pm          float64
    RainToday         object
    Year               int64
    Month              int64
    Day                int64
    dtype: object

<br>

```python
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```

    ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

<br>

```python
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

    ['MinTemp',
     'MaxTemp',
     'Rainfall',
     'Evaporation',
     'Sunshine',
     'WindGustSpeed',
     'WindSpeed9am',
     'WindSpeed3pm',
     'Humidity9am',
     'Humidity3pm',
     'Pressure9am',
     'Pressure3pm',
     'Cloud9am',
     'Cloud3pm',
     'Temp9am',
     'Temp3pm',
     'Year',
     'Month',
     'Day']

<br>

### 수치형 변수들의 결측치 공학

```python
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

    MinTemp            495
    MaxTemp            264
    Rainfall          1139
    Evaporation      48718
    Sunshine         54314
    WindGustSpeed     7367
    WindSpeed9am      1086
    WindSpeed3pm      2094
    Humidity9am       1449
    Humidity3pm       2890
    Pressure9am      11212
    Pressure3pm      11186
    Cloud9am         43137
    Cloud3pm         45768
    Temp9am            740
    Temp3pm           2171
    Year                 0
    Month                0
    Day                  0
    dtype: int64

<br>

```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

    MinTemp            142
    MaxTemp             58
    Rainfall           267
    Evaporation      12125
    Sunshine         13502
    WindGustSpeed     1903
    WindSpeed9am       262
    WindSpeed3pm       536
    Humidity9am        325
    Humidity3pm        720
    Pressure9am       2802
    Pressure3pm       2795
    Cloud9am         10520
    Cloud3pm         11326
    Temp9am            164
    Temp3pm            555
    Year                 0
    Month                0
    Day                  0
    dtype: int64

<br>

```python
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

    MinTemp 0.0044
    MaxTemp 0.0023
    Rainfall 0.01
    Evaporation 0.4283
    Sunshine 0.4775
    WindGustSpeed 0.0648
    WindSpeed9am 0.0095
    WindSpeed3pm 0.0184
    Humidity9am 0.0127
    Humidity3pm 0.0254
    Pressure9am 0.0986
    Pressure3pm 0.0983
    Cloud9am 0.3792
    Cloud3pm 0.4023
    Temp9am 0.0065
    Temp3pm 0.0191

<br>

### **추정**

데이터가 완전히 무작위로 누락되었다고 가정하고 (MCAR), 결측값을 보완하는 데 두 가지 방법을 사용할 수 있다. 하나는 평균 또는 중앙값 보완이고, 다른 하나는 무작위 샘플 보완이다. 데이터 세트에 이상치가 있을 때 중앙값 보완을 사용해야 한다. 따라서, 이번 분석에서는 중앙값 보완을 사용해보자.

결측값 보완은 적절한 통계적 측정값(중앙값)으로 이루어진다. 보완은 학습 데이터 세트에서 이루어져야 하며, 그 후에 테스트 데이터 세트로 전파되어야 한다. 즉, 학습 및 테스트 데이터 세트에서 결측값을 채우기 위해 사용되는 통계 측정값은 학습 세트에서만 추출되어야 한다.

```python
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)

```

<br>

```python
# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustSpeed    0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    Year             0
    Month            0
    Day              0
    dtype: int64

<br>

```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustSpeed    0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    Year             0
    Month            0
    Day              0
    dtype: int64

<br>

모든 결측치가 사라진 것을 볼 수 있다.

<br>

### **범주형 변수의 결측치 공학**

```python
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
```

    Location       0.000000
    WindGustDir    0.065114
    WindDir9am     0.070134
    WindDir3pm     0.026443
    RainToday      0.010013
    dtype: float64

<br>

```python
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```

    WindGustDir 0.06511419378659213
    WindDir9am 0.07013379749283542
    WindDir3pm 0.026443026179299188
    RainToday 0.01001283471350458

<br>

```python
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```

<br>

```python
# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()
```

    Location       0
    WindGustDir    0
    WindDir9am     0
    WindDir3pm     0
    RainToday      0
    dtype: int64

<br>

```python
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
```

    Location       0
    WindGustDir    0
    WindDir9am     0
    WindDir3pm     0
    RainToday      0
    dtype: int64

<br>

마지막으로, `X_train`과 `X_test`의 결측치를 확인해보자.

```python
# check missing values in X_train

X_train.isnull().sum()
```

    Location         0
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustDir      0
    WindGustSpeed    0
    WindDir9am       0
    WindDir3pm       0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    RainToday        0
    Year             0
    Month            0
    Day              0
    dtype: int64

<br>

```python
# check missing values in X_test

X_test.isnull().sum()
```

    Location         0
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustDir      0
    WindGustSpeed    0
    WindDir9am       0
    WindDir3pm       0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    RainToday        0
    Year             0
    Month            0
    Day              0
    dtype: int64

<br>

결측치가 없는 것을 확인할 수 있다.

<br>

### **수치형 변수의 이상치 공학**

`Rainfall`, `Evaporation`, `WindSpeed9am`, `WindSpeed3pm`에서 이상치가 있는 것을 봤었다. 위의 변수에서 최대값을 상한선으로 설정하고 이상치를 제거하기 위해 top-coding 접근 방식을 사용해보자.

```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
```

<br>

```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```

    (3.2, 3.2)

<br>

```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```

    (21.8, 21.8)

<br>

```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```

    (55.0, 55.0)

<br>

```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```

    (57.0, 57.0)

<br>

```python
X_train[numerical].describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>1017.640649</td>
      <td>1015.241101</td>
      <td>4.651801</td>
      <td>4.703588</td>
      <td>16.995062</td>
      <td>21.688643</td>
      <td>2012.759727</td>
      <td>6.404021</td>
      <td>15.710419</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>6.738680</td>
      <td>6.675168</td>
      <td>2.292726</td>
      <td>2.117847</td>
      <td>6.463772</td>
      <td>6.855649</td>
      <td>2.540419</td>
      <td>3.427798</td>
      <td>8.796821</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>980.500000</td>
      <td>977.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.200000</td>
      <td>-5.400000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>1013.500000</td>
      <td>1011.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>12.300000</td>
      <td>16.700000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>1017.600000</td>
      <td>1015.200000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>16.700000</td>
      <td>21.100000</td>
      <td>2013.000000</td>
      <td>6.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>1021.800000</td>
      <td>1019.400000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>21.500000</td>
      <td>26.300000</td>
      <td>2015.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1041.000000</td>
      <td>1039.600000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>40.200000</td>
      <td>46.700000</td>
      <td>2017.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
    </tr>
  </tbody>
</table>
</div>

<br>

`Rainfall`, `Evaporation`, `WindSpeed9am` `WindSpeed3pm` 에서 이상치가 사라진 것을 볼 수 있다.

### **범주형 변수 인코딩**

```python
categorical
```

    ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

<br>

```python
X_train[categorical].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>S</td>
      <td>SSE</td>
      <td>S</td>
      <td>No</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>ENE</td>
      <td>SSE</td>
      <td>SE</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>E</td>
      <td>NE</td>
      <td>N</td>
      <td>No</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>ESE</td>
      <td>SSE</td>
      <td>E</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>W</td>
      <td>N</td>
      <td>SE</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>

<br>

```python
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```

<br>

```python
X_train.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday_0</th>
      <th>RainToday_1</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>S</td>
      <td>41.0</td>
      <td>SSE</td>
      <td>S</td>
      <td>...</td>
      <td>1013.4</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>18.8</td>
      <td>20.4</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>4</td>
      <td>25</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>ENE</td>
      <td>33.0</td>
      <td>SSE</td>
      <td>SE</td>
      <td>...</td>
      <td>1013.1</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>26.4</td>
      <td>27.5</td>
      <td>1</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>E</td>
      <td>31.0</td>
      <td>NE</td>
      <td>N</td>
      <td>...</td>
      <td>1013.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.5</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>ESE</td>
      <td>37.0</td>
      <td>SSE</td>
      <td>E</td>
      <td>...</td>
      <td>1010.8</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>27.3</td>
      <td>29.4</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>10</td>
      <td>30</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>W</td>
      <td>39.0</td>
      <td>N</td>
      <td>SE</td>
      <td>...</td>
      <td>1015.2</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>22.2</td>
      <td>27.0</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>11</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>

<br>

두 개의 추가적인 변수인 `RainToday_0`, `RainToday_1`가 `RainToday` 변수로부터 생긴 것을 볼 수 있다.

`X_train` 훈련세트를 생성해보자.

```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```

<br>

```python
X_train.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>28.0</td>
      <td>65.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>33.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>71.0</td>
      <td>59.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>19.0</td>
      <td>59.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>72.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>

<br>

유사하게, `X_test` 훈련세트도 만들어보자.

```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location),
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```

```python
X_test.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86232</th>
      <td>17.4</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>3.6</td>
      <td>11.1</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>63.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57576</th>
      <td>6.8</td>
      <td>14.4</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>8.5</td>
      <td>46.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>80.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124071</th>
      <td>10.1</td>
      <td>15.4</td>
      <td>3.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>31.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>70.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117955</th>
      <td>14.4</td>
      <td>33.4</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>11.6</td>
      <td>41.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>40.0</td>
      <td>23.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>133468</th>
      <td>6.8</td>
      <td>14.3</td>
      <td>3.2</td>
      <td>0.2</td>
      <td>7.3</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>92.0</td>
      <td>47.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>

<br>

이제 모델 구축을 위한 학습 및 테스트 세트가 준비되었다. 그러나 그 전에, 모든 특성 변수를 동일한 척도로 매핑해야 한다. 이를 `특성 스케일링`<font size="2">feature scaling</font>이라고 합니다. 다음과 같이 수행해보자.

<br>

## **11. 특성 스케일링**

[목차](#목차)

```python
X_train.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>

<br>

```python
cols = X_train.columns
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

```

```python
X_train = pd.DataFrame(X_train, columns=[cols])
```

```python
X_test = pd.DataFrame(X_test, columns=[cols])
```

```python
X_train.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.484406</td>
      <td>0.530004</td>
      <td>0.210962</td>
      <td>0.236312</td>
      <td>0.554562</td>
      <td>0.262667</td>
      <td>0.254148</td>
      <td>0.326575</td>
      <td>0.688675</td>
      <td>0.515095</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.151741</td>
      <td>0.134105</td>
      <td>0.369949</td>
      <td>0.129528</td>
      <td>0.190999</td>
      <td>0.101682</td>
      <td>0.160119</td>
      <td>0.152384</td>
      <td>0.189356</td>
      <td>0.205307</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.375297</td>
      <td>0.431002</td>
      <td>0.000000</td>
      <td>0.183486</td>
      <td>0.565517</td>
      <td>0.193798</td>
      <td>0.127273</td>
      <td>0.228070</td>
      <td>0.570000</td>
      <td>0.370000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.479810</td>
      <td>0.517958</td>
      <td>0.000000</td>
      <td>0.220183</td>
      <td>0.586207</td>
      <td>0.255814</td>
      <td>0.236364</td>
      <td>0.333333</td>
      <td>0.700000</td>
      <td>0.520000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.593824</td>
      <td>0.623819</td>
      <td>0.187500</td>
      <td>0.247706</td>
      <td>0.600000</td>
      <td>0.310078</td>
      <td>0.345455</td>
      <td>0.421053</td>
      <td>0.830000</td>
      <td>0.650000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>

<br>

이제 로지스틱 회귀 분류기에 들어갈 `X_train` 데이터세트가 준비되었다.

<br>

## **12. 모델 학습**

[목차](#목차)

```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)

```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)

<br>

## **13. 결과 예측**

[목차](#목차)

```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

    array(['No', 'No', 'No', ..., 'No', 'No', 'Yes'], dtype=object)

<br>

### **predict_proba 메서드**

**predict_proba** 메서드는 대상 변수(0과 1)에 대한 확률을 배열 형태로 제공합니다.

`0은 비가 오지 않을 확률`을 나타내며, `1은 비가 올 확률`을 나타낸다.

<br>

```python
# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]
```

    array([0.91382428, 0.83565645, 0.82033915, ..., 0.97674285, 0.79855098,
           0.30734161])

<br>

```python
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]
```

    array([0.08617572, 0.16434355, 0.17966085, ..., 0.02325715, 0.20144902,
           0.69265839])

<Br>

## **14. 정확도 점수 확인**

[목차](#목차)

```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

    Model accuracy score: 0.8502

예측된 클래스 레이블을 나타내는 `y_pred_test`와 테스트 세트의 실제 클래스 레이블을 나타내는 `y_test` 이다.

<br>

### **훈련세트와 테스트세트 정확도 비교**

이제 훈련세트와 테스트세트의 과대적합 여부를 확인하기 위해 정확도를 비교해보자.

```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```

    array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)

<br>

```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

    Training-set accuracy score: 0.8476

<br>

### 과대적합과 과소적합 확인

```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

    Training set score: 0.8476
    Test set score: 0.8502

<br>

훈련 세트의 정확도 점수는 0.8476이고, 테스트 세트의 정확도는 0.8501이다. 이 두 값은 꽤 비교가 가능하다. 따라서 과대적합은 아니다.

로지스틱 회귀에서는 C=1의 기본값을 사용한다. 이는 훈련 세트와 테스트 세트 모두 대략 85%의 정확도로 좋은 성능을 제공한다. 그러나 훈련 세트와 테스트 세트 모두 모델의 성능이 매우 비교가 가능하므로, 과소적합일 가능성이 있다.

C를 높이고 더 유연한 모델을 적합시켜보자.

```python
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```

    LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)

<br>

```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

    Training set score: 0.8478
    Test set score: 0.8505

<br>

C값을 100으로 설정한 더 규제가 높은 모델을 사용했을 때, 테스트 세트의 정확도는 0.8266으로 낮아졌다. 이는 모델이 지나치게 단순해져서 일반화 능력이 감소했다는 것을 의미한다. 따라서 C=1보다 더 규제를 가하는 것은 좋지 않은 결과를 가져올 수 있다.

이제 C=0.01로 세팅함으로써 어떤 일이 발생하는 지 봐보자.

```python
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```

    LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)

<br>

```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

    Training set score: 0.8409
    Test set score: 0.8448

<br>

그래서, C=0.01로 더 규제가 된 모델을 사용하면, 기본 매개 변수에 비해 훈련 및 테스트 세트 정확도가 감소한다.

### **null 정확도 비교**

모델의 정확도는 0.8501이다. 하지만, 우리의 모델이 위의 정확도보다 매우 좋다고는 할 수 없다. 그래서 **null 정확도**<font size="2">Null Accuracy</font>를 비교해야만 한다. Null 정확도는 가장 빈번한 클래스를 예측하는 것으로 달성할 수 있는 정확도이다.

따라서 먼저 테스트 세트에서 클래스 분포를 확인해야 한다.

```python
# check class distribution in test set

y_test.value_counts()
```

    No     22067
    Yes     6372
    Name: RainTomorrow, dtype: int64

<br>

가장 빈번한 클래스의 발생 횟수는 22067이다. 따라서 총 발생 횟수로 나누어 22067을 계산하여 null 정확도를 계산할 수 있다.

```python
# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

    Null accuracy score: 0.7759

<br>

우리는 모델의 정확도 점수가 0.8501이지만, 널 정확도 점수는 0.7759임을 알 수 있다. 따라서 우리는 로지스틱 회귀 모델이 클래스 레이블을 예측하는 데 매우 잘하고 있다고 결론을 내릴 수 있다.

<br>

이제 위의 분석을 기반으로 분류 모델의 정확도가 매우 높다고 결론을 내릴 수 있다. 우리 모델은 클래스 레이블을 예측하는 데 아주 좋은 성능을 보인다.

하지만 이 방법은 값의 분포를 보여주지 않으며, 분류기가 어떤 유형의 오류를 발생시키는지 알려주지 않는다.

`오차 행렬`<font size="2">Confusion matrix</font>라는 도구가 도와준다.

<br>

## **15. 오차 행렬**

[목차](#목차)

오차 행렬(<font size="2">Confusion matrix</font>)은 분류 알고리즘의 성능을 요약하는 도구이다. 오차 행렬을 사용하면 분류 모델의 성능과 모델이 생성한 오류 유형을 명확하게 파악할 수 있다. 이는 각 범주별로 올바른 예측과 잘못된 예측을 요약하여 표 형태로 제공한다.

분류 모델의 성능을 평가할 때, 네 가지 종류의 결과가 가능하다:

**True Positives (TP)** - TP는 우리가 관찰이 특정 클래스에 속한다고 예측하고, 실제로 그 클래스에 속하는 경우이다.

**True Negatives (TN)** - TN은 우리가 관찰이 특정 클래스에 속하지 않는다고 예측하고, 실제로 그 클래스에 속하지 않는 경우이다.

**False Positives (FP)** - FP는 우리가 관찰이 특정 클래스에 속한다고 예측했지만, 실제로는 그 클래스에 속하지 않는 경우이다. 이러한 종류의 오류를 `Type I error`라고 한다.

**False Negatives (FN)** - FN은 우리가 관찰이 특정 클래스에 속하지 않는다고 예측했지만, 실제로는 그 클래스에 속하는 경우이다. 이러한 오류는 매우 심각하며, `Type II error`라고 하다.

이 네 가지 결과는 다음과 같이 요약되며, 아래의 오차 행렬에서 확인할 수 있다.

<br>

```python
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

    Confusion matrix

     [[20892  1175]
     [ 3086  3286]]

    True Positives(TP) =  20892

    True Negatives(TN) =  3286

    False Positives(FP) =  1175

    False Negatives(FN) =  3086

<br>

이 오차 행렬은 20892 + 3285 = 24177개의 올바른 예측과 3087 + 1175 = 4262개의 틀린 예측을 보여줍니다.

다음과 같은 결과를 얻을 수 있다.

`True Positives` (실제 Positive:1 및 예측 Positive:1) - 20892

`True Negatives` (실제 Negative:0 및 예측 Negative:0) - 3285

`False Positives` (실제 Negative:0이지만 예측 Positive:1) - 1175 `(Type I error)`

`False Negatives` (실제 Positive:1이지만 예측 Negative:0) - 3087 `(Type II error)`

<br>

```python
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7f28b1306208>

![image](https://user-images.githubusercontent.com/106001755/231873438-8d52c702-dbe8-4a3a-af98-a87d72d9ab47.png)

<br>

## **16. 분류 지표**

[목차](#목차)

### **분류 보고서**

**분류 보고서**는 분류 모델의 성능을 평가하는 또 다른 방법이다. 이 보고서는 모델의 **정밀도**(precision), **재현율**(recall), **F1** 점수와 **지원**(support) 점수를 표시한다.

분류 보고서는 다음과 같이 출력할 수 있습니다:-

<br>

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

                  precision    recall  f1-score   support

              No       0.87      0.95      0.91     22067
             Yes       0.74      0.52      0.61      6372

        accuracy                           0.85     28439
       macro avg       0.80      0.73      0.76     28439
    weighted avg       0.84      0.85      0.84     28439

<br>

### **분류 정확성**

```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```

```python
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

```

    Classification accuracy : 0.8502

<br>

### **분류 오류**

```python
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

```

    Classification error : 0.1498

<br>

### **정밀도**

정밀도는 예측된 양성 결과 중에서 올바르게 예측된 결과의 비율로 정의할 수 있다. 이것은 참 양성(True Positive; TP)을 예측한 비율이다.

수학적으로, 정밀도는 TP를 (TP + FP)로 나눈 비율로 정의된다.

```python
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

```

    Precision : 0.9468

<br>

### **재현율**

재현율은 참 양성 중에서 올바르게 예측된 양성의 비율을 나타냅니다. 참 양성(TP)를 참 양성과 거짓 음성의 합인 (TP + FN)으로 나눈 비율로 표현할 수 있다. 재현율은 민감도(Sensitivity)라고도 불립니다.

수학적으로는 TP / (TP + FN)으로 정의할 수 있다.

```python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

    Recall or Sensitivity : 0.8713

<br>

### **참 양성 비율**

참 양성 비율은 재현율과 같은 말이다.

```python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

    True Positive Rate : 0.8713

<br>

### **거짓 양성 비율**

```python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

    False Positive Rate : 0.2634

<br>

### **특이도**

```python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

    Specificity : 0.7366

<br>

### **f1-score**

f1-score는 정밀도와 재현율의 가중 조화 평균이다. 최고의 f1-score는 1.0이며 최악의 경우는 0.0이다. f1-score는 정확도 측정치보다 항상 낮다. f1-score의 가중 평균은 전역 정확도보다 분류기 모델을 비교하는 데 사용해야 한다.

<br>

### Support

Support는 데이터셋에서 해당 클래스가 실제로 발생한 횟수를 나타내는 값이다.

<br>

## **17. 임계값 조정**

[목차](#목차)

```python
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

    array([[0.91382428, 0.08617572],
           [0.83565645, 0.16434355],
           [0.82033915, 0.17966085],
           [0.99025322, 0.00974678],
           [0.95726711, 0.04273289],
           [0.97993908, 0.02006092],
           [0.17833011, 0.82166989],
           [0.23480918, 0.76519082],
           [0.90048436, 0.09951564],
           [0.85485267, 0.14514733]])

<br>

### **관찰 결과**

- 각 행(row)의 숫자는 1에 합산된다.

- 2개의 열(column)은 0과 1의 2개의 클래스를 나타낸다.

  - 클래스 0 - 내일 비가 오지 않을 확률에 대한 예측 확률.

  - 클래스 1 - 내일 비가 올 확률에 대한 예측 확률.

- 예측 확률의 중요성

  - 우리는 비 또는 비가 오지 않을 확률에 따라 관찰 값을 순위 지정할 수 있다.

- predict_proba 과정

  - 확률 값을 예측한다.

  - 가장 높은 확률을 가진 클래스를 선택한다.

- 분류 임계값

  - 분류 임계값이 0.5

  - 클래스 1 - 확률 > 0.5 이면 비가 올 확률이 예측된다.

  - 클래스 0 - 확률 < 0.5 이면 비가 오지 않을 확률이 예측된다.

<br>

```python
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prob of - No rain tomorrow (0)</th>
      <th>Prob of - Rain tomorrow (1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.913824</td>
      <td>0.086176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835656</td>
      <td>0.164344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.820339</td>
      <td>0.179661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.990253</td>
      <td>0.009747</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.957267</td>
      <td>0.042733</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.979939</td>
      <td>0.020061</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.178330</td>
      <td>0.821670</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.234809</td>
      <td>0.765191</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.900484</td>
      <td>0.099516</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.854853</td>
      <td>0.145147</td>
    </tr>
  </tbody>
</table>
</div>

<br>

```python
# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]
```

    array([0.08617572, 0.16434355, 0.17966085, 0.00974678, 0.04273289,
           0.02006092, 0.82166989, 0.76519082, 0.09951564, 0.14514733])

<br>

```python
# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg.predict_proba(X_test)[:, 1]
```

```python
# plot histogram of predicted probabilities


# adjust the font size
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

    Text(0, 0.5, 'Frequency')

![image](https://user-images.githubusercontent.com/106001755/231873492-b87ba637-eeab-4da4-9837-c1d729453532.png)

<br>

### **관찰 결과**

- 위의 히스토그램은 매우 양으로 치우쳤다.

- 첫번째 열은 약 0.0부터 0.1 사이의 확률을 가진 약 15000개의 관측치가 있다는 것을 보여준다.

- 확률이 0.5보다 큰 일부 관측치가 존재한다.

- 이 작은 수의 관측치는 내일 비가 올 것이라고 예측한다.

- 대부분의 관측치는 내일 비가 오지 않을 것이라고 예측한다.

<br>

### **더 낮은 임계값**

```python
from sklearn.preprocessing import binarize

for i in range(1,5):

    cm1=0

    y_pred1 = logreg.predict_proba(X_test)[:,1]

    y_pred1 = y_pred1.reshape(-1,1)

    y_pred2 = binarize(y_pred1, i/10)

    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')

    cm1 = confusion_matrix(y_test, y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',

            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n',

            cm1[0,1],'Type I errors( False Positives), ','\n\n',

            cm1[1,0],'Type II errors( False Negatives), ','\n\n',

           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',

           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',

           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',

            '====================================================', '\n\n')
```

    With 0.1 threshold the Confusion Matrix is

     [[12726  9341]
     [  547  5825]]

     with 18551 correct predictions,

     9341 Type I errors( False Positives),

     547 Type II errors( False Negatives),

     Accuracy score:  0.6523084496641935

     Sensitivity:  0.9141556811048337

     Specificity:  0.5766982371867494

     ====================================================


    With 0.2 threshold the Confusion Matrix is

     [[17066  5001]
     [ 1234  5138]]

     with 22204 correct predictions,

     5001 Type I errors( False Positives),

     1234 Type II errors( False Negatives),

     Accuracy score:  0.7807588171173389

     Sensitivity:  0.8063402385436284

     Specificity:  0.7733720034440568

     ====================================================


    With 0.3 threshold the Confusion Matrix is

     [[19080  2987]
     [ 1872  4500]]

     with 23580 correct predictions,

     2987 Type I errors( False Positives),

     1872 Type II errors( False Negatives),

     Accuracy score:  0.8291430781673055

     Sensitivity:  0.7062146892655368

     Specificity:  0.8646395069560883

     ====================================================


    With 0.4 threshold the Confusion Matrix is

     [[20191  1876]
     [ 2517  3855]]

     with 24046 correct predictions,

     1876 Type I errors( False Positives),

     2517 Type II errors( False Negatives),

     Accuracy score:  0.845529027040332

     Sensitivity:  0.6049905838041432

     Specificity:  0.9149861784565188

     ====================================================

<br>

### **Comments**

- 이진 분류 문제에서는 기본적으로 예측된 확률을 클래스 예측으로 변환하기 위해 0.5의 임계값을 사용한다.

- 임계값은 민감도 또는 특이도를 높이기 위해 조정할 수 있다.

- 민감도와 특이도는 서로 역의 관계를 가지고 있습니다. 한 쪽을 높이면 다른 쪽은 낮아지게 된다(트레이드 오프 관계).

- 임계값을 높이면 정확도가 높아지는 것을 볼 수 있다.

- 임계값 조정은 모델 구축 프로세스에서 마지막 단계 중 하나여야 한다.

<br>

## **18. ROC-AUC**

[목차](#목차)

### **ROC 곡선**

분류 모델의 성능을 시각적으로 측정하는 또 다른 도구는 ROC 곡선이다.

ROC 곡선은 다양한 임계값에서 참 양성 비율(TPR)와 거짓 양성 비율(FPR)을 나타냅니다.

다양한 임계값에서의 TPR과 FPR로 이루어진 ROC 곡선의 일반적인 성능을 파악할 수 있다. 임계값을 낮출수록, 더 많은 항목이 긍정적으로 분류될 수 있다.

<br>

```python
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

```

![image](https://user-images.githubusercontent.com/106001755/231873524-30b35f73-1978-4be5-a354-f9b46fd957e0.png)

### **ROC-AUC**

ROC AUC는 분류기의 성능을 비교하는 기술이다. 이 기술에서는 곡선 아래 면적인 AUC (Area Under Curve)를 측정한다. 완벽한 분류기는 ROC AUC가 1이 되고, 완전한 무작위 분류기는 ROC AUC가 0.5가 된다.

따라서, ROC AUC는 ROC 그래프에서 곡선 아래 부분의 백분율을 나타낸다.

<br>

```python
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

    ROC AUC : 0.8729

<br>

### Comments

- ROC AUC는 값이 높을수록 분류기의 성능이 더 좋다.

<br>

```python
# calculate cross-validated ROC AUC

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

    Cross validated ROC AUC : 0.8695

<br>

## **19. k-Fold 교차 검증**

[목차](#목차)

```python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

    Cross-validation scores:[0.84686387 0.84624852 0.84633642 0.84963298 0.84773626]

<br>

교차 검증 정확도는 평균을 계산하여 요약할 수 있다.

```python
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

    Average cross-validation score: 0.8474

<br>

원래 모델의 점수는 0.8476으로 나타났다. 교차 검증의 평균 점수는 0.8474다. 따라서, 교차 검증이 성능 향상을 가져오지 않는다는 결론을 내릴 수 있다.

<br>

## **20. 그리드 탐색 CV를 사용한 하이퍼파라미터 최적화**

[목차](#목차)

```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']},
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)

```

    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                              fit_intercept=True,
                                              intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='warn',
                                              n_jobs=None, penalty='l2',
                                              random_state=0, solver='liblinear',
                                              tol=0.0001, verbose=0,
                                              warm_start=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'penalty': ['l1', 'l2']}, {'C': [1, 10, 100, 1000]}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)

<br>

```python
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```

    GridSearch CV best score : 0.8474


    Parameters that give the best results :

     {'penalty': 'l1'}


    Estimator that was chosen by the search :

     LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l1',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)

<br>

```python
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

    GridSearch CV score on test set: 0.8507

<br>

### Comments

원래 모델의 테스트 정확도는 0.8501이고, GridSearch CV의 정확도는 0.8507다.

따라서, 이 모델에서 GridSearch CV가 성능을 개선시켰다는 것을 알 수 있다.

<br>

## **21. 결과 및 결론**

[목차](#목차)

1. 로지스틱 회귀 모델의 정확도 점수는 0.8501다. 따라서 이 모델은 오늘 오스트레일리아에서 비가 올 것인지 아닌지 예측하는 데 아주 잘 작동한다.

2. 일부 관측치는 내일 비가 올 것으로 예측하고 있지만, 대부분의 관측치는 내일 비가 오지 않을 것으로 예측하고 있다.

3. 모델은 과대 적합의 흔적이 없다.

4. C값을 높일수록 테스트 세트 정확도와 약간 증가한 훈련 세트 정확도를 얻을 수 있다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘할 것으로 결론을 내릴 수 있다.

5. 임계값을 높이면 정확도가 증가한다.

6. 모델의 ROC AUC는 1에 가까워진다. 따라서 이 분류기는 내일 비가 올지 안 올지 예측하는 데 아주 잘 작동한다고 결론을 내릴 수 있다.

7. 원래 모델의 정확도 점수는 0.8501이고, RFECV 이후 정확도 점수는 0.8500다. 따라서 약간의 feature를 줄여서 비슷한 정확도를 얻을 수 있다.

8. 원래 모델에서 FP = 1175이고, FP1 = 1174다. 거짓 양성의 수가 거의 같다. 또한, FN = 3087이고, FN1 = 3091다. 거짓 음성이 약간 더 높아졌다.

9. 원래 모델의 점수는 0.8476다. 교차 검증의 평균 점수는 0.8474다. 따라서 교차 검증은 성능 향상을 가져오지 않는다는 결론을 내릴 수 있다.

10. 원래 모델의 테스트 정확도는 0.8501이고, GridSearch CV의 정확도는 0.8507다. GridSearch CV가 이 모델에서 성능을 개선시켰다는 것을 알 수 있다.

## **22. 참고 문헌**

[목차](#목차)

1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron

2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido

3. Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves

4. Udemy course – Feature Engineering for Machine Learning by Soledad Galli

5. Udemy course – Feature Selection for Machine Learning by Soledad Galli

6. https://en.wikipedia.org/wiki/Logistic_regression

7. https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html

8. https://en.wikipedia.org/wiki/Sigmoid_function

9. https://www.statisticssolutions.com/assumptions-of-logistic-regression/

10. https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

11. https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression

12. https://www.ritchieng.com/machine-learning-evaluate-classification-model/
