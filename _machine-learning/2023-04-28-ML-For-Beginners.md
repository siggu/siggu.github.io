---
layout: single
title: "ML Model을 사용해 Web App 만들기"
category: machine-learning
tags: ["머신러닝", "웹앱"]
---

# ML Model을 사용해 Web App 만들기

[ML-For-Beginners](<[https://github.com/microsoft/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/README.md](https://github.com/microsoft/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/README.md)>)에 있는 내용을 통해 Web App을 만들어볼 것이다.

이 강좌에서는 NUFORC 데이터베이스에서 가져온 지난 백 년간의 UFO 목격 보고서를 기반으로 ML 모델을 훈련시키는 방법에 대해 다룬다.

다음과 같은 내용을 학습한다.

- 훈련된 모델을 ‘pickle’ 하는 방법
- Flask 앱에서 해당 모델을 사용하는 방법

데이터 정제와 모델 훈련에 대한 내용은 여전히 노트북을 사용하여 진행하지만, 웹 앱에서 모델을 사용하는 것과 같이 모델을 실제로 활용하는 과정도 함께 탐구한다.

이를 위해서, Flask를 이용해 웹 앱을 구축해야 한다.

<br>

## 강의 전 퀴즈

### **앱 만들기**

머신 러닝 모델을 사용하기 위한 웹 앱을 만드는 방법은 여러 가지가 있다. 웹 구조에 따라 모델 훈련 방법이 영향을 받을 수 있다. 데이터 과학 그룹에서 모델을 훈련시켰으며, 이를 앱에서 사용하도록 요청하는 비즈니스에서 일하고 있다고 해보자.

<br>

**고려사항**

아래와 같은 많은 질문들이 필요하다:

- **Is it a web app or a mobile app?** If you are building a mobile app or need to use the model in an IoT context, you could use [TensorFlow Lite](https://www.tensorflow.org/lite/) and use the model in an Android or iOS app.
- **Where will the model reside?** In the cloud or locally?
- **Offline support.** Does the app have to work offline?
- **What technology was used to train the model?** The chosen technology may influence the tooling you need to use.
  - **Using TensorFlow.** If you are training a model using TensorFlow, for example, that ecosystem provides the ability to convert a TensorFlow model for use in a web app by using [TensorFlow.js](https://www.tensorflow.org/js/).
  - **Using PyTorch.** If you are building a model using a library such as [PyTorch](https://pytorch.org/), you have the option to export it in [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format for use in JavaScript web apps that can use the [Onnx Runtime](https://www.onnxruntime.ai/). This option will be explored in a future lesson for a Scikit-learn-trained model.
  - **Using Lobe.ai or Azure Custom Vision.** If you are using an ML SaaS (Software as a Service) system such as [Lobe.ai](https://lobe.ai/) or [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) to train a model, this type of software provides ways to export the model for many platforms, including building a bespoke API to be queried in the cloud by your online application.

결국 우리는 Python을 기반으로 한 노트북을 사용해 왔으므로, 이러한 노트북에서 훈련된 모델을 Python 기반 웹 앱에서 읽을 수 있는 형식으로 내보네는 데 필요한 단계를 살펴볼 것이다.

<br>

## 도구

Flask와 Pickle은 모두 파이썬에서 실행되는 도구다.

- [Flask](https://palletsprojects.com/p/flask/)는 생성자들에 의해 ‘마이크로 프레임워크’로 정의되며, Python을 사용하여 웹 프레임워크의 기본 기능을 제공하고 템플릿 엔진을 사용하여 웹 페이지를 구축할 수 있다.
- [Pickle](https://docs.python.org/3/library/pickle.html)🥒은 파이썬 객체 구조를 직렬화하고, 역직렬화하는 파이썬 모듈이다. 모델을 ‘pickle’할 때는 웹에서 사용할 수 있도록 직렬화하거나 평면화한다. 주의해야 할 점은 pickle이 본질적으로 안전하지 않기 때문에 ‘un-pickle’ 파일을 복원할 때 주의해야 한다. pickled 파일은 확장자 `.pkl`을 가진다.

<br>

## 연습 - 데이터 정리

이번 수업에서는 [NUFORC](https://nuforc.org/) (The National UFO Reporting Center)에서 수집한 80,000건의 UFO 목격담 데이터를 사용한다. 이 데이터에는 다음과 같이 흥미로운 UFO 목격담이 포함되어 있다.

- **긴 예시**: “한 남자가 밤에 초원에 비춰지는 빔에서 나와 Texas Instruments 주차장 쪽으로 달려간다.”
- **짧은 예시**: "빛들이 우리를 추적했다.”

[ufos.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/data/ufos.csv) 스프레드시트는 목격이 발생한 `도시`, `주` 및 `국가` , 물체의 `모양`과 `위도` 및 `경도` 와 같은 열이 포함되어 있다.

1. 빈 노트북에서 pandas, matplotlib 및 numpy를 import 하여 ufos 스프레드시트를 가져온다. 샘플 데이터셋을 확인할 수 있다.

   ```python
   import pandas as pd
   import numpy as np

   ufos = pd.read_csv('./data/ufos.csv')
   ufos.head()
   ```

2. ufos 데이터를 새 타이틀을 가진 작은 데이터프레임으로 변환한다. `Country` 필드의 고유값을 확인해보자.

   ```python
   ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

   ufos.Country.unique()
   ```

3. 이제 결측치가 있는 데이터를 제거하고 1-60초 사이에 발생한 관측만 가져와서 처리할 필요가 있다.

   ```python
   ufos.dropna(inplace=True)

   ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

   ufos.info()
   ```

4. 사이킷런의 `LabelEncoder` 라이브러리를 가져와 국가에 대한 텍스트 값을 숫자로 변환해보자.

   - `LabelEncoder`는 데이터를 알파벳순으로 인코딩해준다.

   ```python
   from sklearn.preprocessing import LabelEncoder

   ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

   ufos.head()
   ```

데이터의 생김새를 확인할 수 있다.

```python
  Seconds	Country	Latitude	Longitude
2	20.0	3		53.200000	-2.916667
3	20.0	4		28.978333	-96.645833
14	30.0	4		35.823889	-80.253611
23	60.0	4		45.582778	-122.352222
24	3.0		3		51.783333	-0.783333
```

<br>

## 연습 - 모델 구축

이제 데이터를 훈련셋과 테스트셋으로 나눌 준비가 되었다.

1. X 벡터로 사용할 세 가지 특성을 선택하고, y 벡터는 `Country`가 된다. `Seconds`, `Latitude` 및 `Longitude`를 입력하면 국가 ID를 반환할 수 있도록 해보자.

   ```python
   from sklearn.model_selection import train_test_split

   Selected_features = ['Seconds','Latitude','Longitude']

   X = ufos[Selected_features]
   y = ufos['Country']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   ```

1. 로지스틱 회귀를 사용해 모델을 훈련시켜보자.

   ```python
   from sklearn.metrics import accuracy_score, classification_report
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)

   print(classification_report(y_test, predictions))
   print('Predicted labels: ', predictions)
   print('Accuracy: ', accuracy_score(y_test, predictions))
   ```

정확도는 약 95%로 나쁘지 않다. `Country`와 `Latitude`/`Logitude`에 상관관계가 있기 때문이다.

모델은 혁신적이지는 않다. 위도와 경도에서 국가를 추론할 수 있어야 하기 때문이다. 하지만, 깨끗하게 정리하고 내보낸 원시 데이터에서 훈련시켜 이 모델을 웹 앱에서 사용하는 것은 좋은 연습이다.

<br>

## 연습 - 모델 ‘pickle’하기

이제 모델을 pickle로 저장할 차례다. 이를 위해 몇 줄의 코드가 필요하다.

pickle로 저장된 모델을 불러와, 초(sec), 위도(latitude) 및 경도(longitude) 값을 담은 샘플 데이터 배열을 사용해 모델을 테스트해보자.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

모델은 3을 반환했다. 이는 영국의 국가 코드다.

<br>

## 연습 - Flask app 구축하기

이제 Flask 앱을 구축하여 모델을 호출하고 비슷한 결과를 반환하지만 시각적으로 더 매력적인 방식으로 표시할 수 있다.

1. ufo-model.pkl 파일이 있는 notebook.ipynb 파일 옆에 web-app 이라는 이름의 폴더를 만든다.

1. 그 폴더 안에 static 이라는 폴더를 만들고 그 안에 css 라는 폴더를 만든다. 그러면 다음과 같은 파일과 디렉터리가 있어야 한다.

```python
web-app/
  static/
    css/
  templates/
notebook.ipynb
ufo-model.pkl
```

3. web-app 폴더를 만들기 위한 첫 파일은 requirements.txt 파일이다. 자바스크립트 앱의 package.json과 같이, 이 파일 리스트는 앱에서 필요한 의존성을 나열하는 파일이다. requirements.txt에 다음 라인을 추가하자.

```python
scikit-learn
pandas
numpy
flask
```

4. 이제 web-app으로 이동해 이 파일을 실행해보자.

```python
cd web-app
```

5. 터미널에서 requirements.txt에 나열된 라이브러리를 설치하려면 pip install을 실행하자.

```python
pip install -r requirements.txt
```

6. 앱을 완성하기 위해 다음 세 파일을 생성하자.

   - root 디렉터리에 app.py 생성
   - templates 디렉터리에 index.html 파일 생성
   - static/css 디렉터리에 styles.css 파일 생성

7. style.css 파일을 작성해보자.

```python
body {
	width: 100%;
	height: 100%;
	font-family: 'Helvetica';
	background: black;
	color: #fff;
	text-align: center;
	letter-spacing: 1.4px;
	font-size: 30px;
}

input {
	min-width: 150px;
}

.grid {
	width: 300px;
	border: 1px solid #2d2d2d;
	display: grid;
	justify-content: center;
	margin: 20px auto;
}

.box {
	color: #fff;
	background: #2d2d2d;
	padding: 12px;
	display: inline-block;
}
```

8. index.html 파일을 작성해보자.

```python
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>🛸 UFO Appearance Prediction! 👽</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
  </head>

  <body>
    <div class="grid">

      <div class="box">

        <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>

        <form action="{{ url_for('predict')}}" method="post">
          <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
          <input type="text" name="latitude" placeholder="Latitude" required="required" />
          <input type="text" name="longitude" placeholder="Longitude" required="required" />
          <button type="submit" class="btn">Predict country where the UFO is seen</button>
        </form>

        <p>{{ prediction_text }}</p>

      </div>

    </div>

  </body>
</html>
```

이 파일의 템플릿을 살펴보자. 예측 텍스트와 같이 앱에서 제공될 변수 주변의 'mustache' 구문 {{}}에 주목할 필요가 있다. /predict 라우트에 예측을 게시하는 폼도 있다.

마지막으로, 모델의 사용과 예측 표시를 구동하는 Python 파일을 작성할 준비가 되었다.

9. app.py에 내용을 추가해보자.

```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./ufo-model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )

if __name__ == "__main__":
    app.run(debug=True)
```

<aside>
💡 Flask를 사용하여 웹 앱을 실행할 때 debug=True를 추가하면 애플리케이션에 대한 변경 사항이 서버를 다시 시작할 필요 없이 즉시 된다.. 하지만 이 모드를 프로덕션 앱에서는 사용하면 안된다.

</aside>

<br>

만약 python app.py 또는 python3 app.py를 실행하면, 로컬로 웹 서버가 시작되고, UFO가 목격된 장소에 관한 궁금증에 대한 답변을 얻기 위해 짧은 폼을 작성할 수 있다.

![image](https://user-images.githubusercontent.com/106001755/235427694-6504240e-3067-42c1-a3ff-1eeb60e6c9ca.png)

우리나라 위도 경도를 입력해봤다.

![image](https://user-images.githubusercontent.com/106001755/235427546-73f4f7b5-4990-411e-ab3a-ddf5e1a735ff.png)

결과는 독일이 나왔다...
