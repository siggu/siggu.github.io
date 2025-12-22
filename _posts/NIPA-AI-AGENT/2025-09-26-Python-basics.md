---
layout: single
title: "파이썬 기초"
categories: [nipa-ai-agent]
subcategory: Python
---

> 2025-09-26 nipa-ai-agent-파이썬-기초

## 파이썬 가상 환경 구성

### 리눅스 서버(Xshell)에 파이썬 가상 환경 구성

### 개인 서버

```bash
# 파이썬 가상환경 기본 패키지 설치
sudo apt install python3-venv
sudo apt install python3-pip
```

### 공용 서버

> 위 기본 패키지는 이미 설치되어 있다고 가정

```bash
# 가상 환경 생성
python3 –m venv [가상_환경_이름]

# 가상 환경 활성
source [가상_환경_이름]/bin/activate

# python 실행
python

# 가상 환경 해제
deactivate
```

> **앞으로 [개인서버] 라고 붙이지 않는 이상 공용 서버에서 진행**

`~~/tmp/anacondadownload.sh` 이 파일을 폴더 하나 만들어서 복사~~

```bash
(base) edu007@vultr:~/sw$ cp /tmp/anacondadownload.sh .
```

`~~bash` 사용해서 다운로드~~

```bash
(base) edu007@vultr:~/sw$ bash anacondadownload.sh
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 1051M  100 1051M    0     0   225M      0  0:00:04  0:00:04 --:--:--  222M
```

## 기타 필요한 명령어들

```bash
# 현재 anaconda 환경에서 생성된 가상환경 목록
conda info --envs

# 가상 환경 활성화
conda activate [가상_환경_이름]

# 가상 환경 비활성화
conda deactivate

# 생성된 가상 환경 삭제
conda remove --name [가상_환경_이름] -all

# 현재 가상 환경에 필요한 패키지를 추가 설치
pip install [패키지명]
```

## PC 환경에서 파이썬 가상 환경 구성

(1) https://www.anaconda.com/download 에서 anaconda 설치파일 download

(2) Anaconda3-2024.10-1-Windows-x86_64.exe를 실행

(3) 윈도우 검색에서 “anaconda” 를 입력 후, “anaconda Prompt”를 실행

(4) anaconda Prompt창에서 python을 입력하여 python 이 정상 설치되었는지 확인

(5) 아래 명령을 입력하여 가상환경을 하나 생성한다. (파이썬버전은 3.10으로 한다.)

```bash
conda create –n 가상환경이름 python=파이썬버전
```

## 파이썬 기초

### 기본 문법

**변수 선언(타입 지정 불필요)**

```bash
a = 10
b = 3.14
c = "Python"
d = True
print(type(a))  # <class 'int'>
```

**연산자**

- 산술 연산자: //, %, \*\*, …
- 할당 연산자: +=, -=, …
- 비교 연산자: is, is not(객체 식별)
- 멤버십 연산자: in, not in
- 비트 연산자: &, |, ^, ~, <<, >>

```python
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a == b)  # True
print(a is b)  # True
print(a == c)  # True
print(a is c)  # False
```

```python
print(5 / 2)   # 2.5
print(5 // 2)  # 2
print(2 ** 3)  # 8

```

**문자열 처리**

```python
text = "Hello, Python!"
print(text.strip())            # "Hello, Python!"
print(text.replace("H", "J"))  # "Jello, Python!"
print("," in text)             # True

name = "Bob"
age = 25
print(f"{name} is {age} years old")  # Bob is 25 years old
```

## 제어 구조

**조건문**

```python
score = int(input())

if score >= 90:
  print("A")
elif score >= 80:
  print("B")
else:
  print("C")
```

**반복문**

```python
for i in range(5):
  print(i)

fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
  print(fruit)

count = 0
while count < 5:
  print(count)
  count += 1
```

**comprehension, enumerate**

```python
# list comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]
print(squares)    # [0, 4, 16, 36, 64]

# enumerate
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits, start=1):
  print(f"{idx}. {fruit}")
```

## 자료구조

```python
numbers = [1, 2, 3, 4, 5]
numbers.append(6)        # [1, 2, 3, 4, 5, 6]
numbers.insert(2, 10)    # [1, 2, 10, 3, 4, 5, 6]
numbers.remove(3)        # [1, 2, 10, 4, 5, 6]
last = numbers.pop()     # [1, 2, 10, 4, 5]

print(numbers[1:3])      # [2, 10]
```

```python
point = (10, 20)
x, y = point      # 언패킹
print(x, y)       # 10, 20
```

```python
a = {1, 2, 3}
b = {3, 4, 5}

print(a | b)    # {1, 2, 3, 4, 5}
print(a & b)    # {3}
```

```python
student = {"name": "태형", "age": 21, "major": "컴퓨터공학"}

print(student["name"])      # 태형
print(student["age"])       # 21

student["grade"] = "A"      # 추가
student["age"] = 21         # 수정

for key in student:
  print(f"{key}: {student[key]}")
```

**얕은 복사, 깊은 복사**

```python
# 얕은 복사
original = [[1, 2], [3, 4]]
copied = original.copy()
copied[0][0] = 99
print(original)    # [[99, 2], [3, 4]] -> 문제!

# 깊은 복사
import copy
original = [[1, 2], [3, 4]]
deep_copied = copy.deepcopy(original)
deep_copied[0][0] = 99
print(original)    # [[1, 2], [3, 4]] -> 문제 x
```

## 함수

```python
def greet(name, message="안녕하세요."):
	return f"{message}, {name}님."

print(greet("철수"))                  # 안녕하세요, 철수님.
print(greet("영희", "반갑습니다."))   # 반갑습니다, 영희님.
```

```python
def add(a, b):
  return a + b

def sub(a, b):
  return a - b

print("덧셈 결과: ", add(10, 5))      # 덧셈 결과: 15
print("뺄셈 결과: ", sub(10, 5))      # 뺄셈 결과: 5
```

```python
def total_sum(*args):
  return sum(args)

print(total_sum(1, 2, 3, 4, 5))    # 15
```

## File

```python
# 파일 입출력
with open("example.txt", "w", encoding="utf-8") as f:
  f.write("Python 파일 입출력 테스트\n")

with open("example.txt", "r", encoding="utf-8") as f:
  content = f.read()
  print(content)
```

```python
# json 입출력
import json

data = {"name": "Alice", "age": 25}

with open("data.json", "w") as f:
  json.dump(data.f)

with open("data.json", "r") as f:
  loaded = json.load(f)
  print(loaded)     # {'name': 'Alice', 'age': 25}
```

## 객체

```python
# 객체 선언 및 접근
class Student:
  def __init__(self, name):
    self.name = name
    self.scores = []

  def add_score(self, score):
    self.scores.append(score)

  def average(self):
    return sum(self.scores) / len(self.scores)

s = Student("나")

tests = ["국어", "영어", "수학", "과학"]

for test in tests:
  s.add_score(int(input(f"{test}의 시험 점수를 입력해 주세요.\n")))

print(f"{s.name}의 평균  시험 점수는 {s.average()}점 입니다.")
```

```python
# 클래스 변수와 인스턴스 변수
class Student:
  count = 0      # 클래스 변수 (모든 인스턴스 공유)

  def __init__(self, name):
    self.name = name
    Student.count += 1

  def show(self):
    print(f"{self.name} 학생입니다.")

  s1 = Student("민지")
  s2 = Student("지후")

  print("총 학생 수: ", Student.count)    # 총 학생 수: 2
```

```python
# 상속
class Animal:
  def __init__(self, name):
    self.name = name

  def speak(self):
    print(f"{self.name}은 ... 소리를 낸다.")

class Dog(Animal):
  def speak(self):
    print(f"{self.name}가 멍멍 짖는다.")

class Cat(Animal):
  def speak(self):
    print(f"{self.name}가 야옹 울다.")

d = Dog("바둑이")
c = Cat("나비")
e = Animal("거북이")
d.speak()
c.speak()
e.speak()

```

## 소켓

### 개인 ↔ 개인, 공용 ↔ 공용

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 5007))
server_socket.listen(1)

print("Echo Server Started!")
client_socket, addr = server_socket.accept()
print(f"Client connected: {addr}")

while True:
  # 클라이언트로부터 데이터를 바이트 형태로 받음
  data = client_socket.recv(1024)
  if not data:
    break

  # 받은 데이터를 문자열로 변환 (디코딩)
  received_msg = data.decode()
  print(f"Received: {received_msg}")

  # 응답할 메시지를 문자열 형태로 만듦
  msg_to_send = f"from server -> {received_msg}"

  # 응답 메시지를 바이트로 변환하여(인코딩) 클라이언트에 전송
  client_socket.send(msg_to_send.encode())

client_socket.close()
server_socket.close()
```

```python
import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 5007))

while True:
  msg = input("보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.): \n")
  if msg in ["q", "quit", ""]:
    break

  client_socket.send(msg.encode())
  data = client_socket.recv(1024)
  print(f"Echo: {data.decode()}")

client_socket.close()
```

`localhost`라고 하면 개인↔개인, 공용↔공용 끼리만 된다.

Xshell 창을 두 개 연 다음, 한 쪽에는 `echo_server.py`, 다른 한 쪽에는 `echo_client.py`를 실행시킨다.

> server → clinet 순서로 실행시켜야 함.

- 파일 보내기
  ```bash
  (myvenv) edu007@vultr:~/work/test$ scp -i ~/.ssh/ssh-key-2025-09-24-007.key /home/edu007/work/test/simple_echo_server.py [ubuntu@161.118.158.91](mailto:ubuntu@161.118.158.91):~/work/test/
  simple_echo_server.py                                     100%  535   216.2KB/s   00:00
  ```
  ```bash
  (myvenv) edu007@vultr:~/work/test$ scp -i ~/.ssh/ssh-key-2025-09-24-007.key /home/edu007/work/test/simple_echo_client.py [ubuntu@161.118.158.91](mailto:ubuntu@161.118.158.91):~/work/test/
  simple_echo_client.py                                     100%  400   154.4KB/s   00:00
  ```
- 파일들을 한 번에 보내는 법
  보내는 파일명을 적을 때 simple_echo\*.py 라고 하면 simple_echo로 시작하는 모든 파일을 보냄

---

## 개인 ↔ 서버

- 개인 서버에서 방화벽 허용
  ```bash
  ubuntu@edu007-20250924-1450:~/work/test$ sudo iptables -I INPUT 1 -p tcp --dport 5000:9000 -j ACCEPT
  ```
  **한 줄 요약**
  : "외부에서 이 서버로 들어오는 TCP 방식의 요청 중, 목적지 포트가 5000번부터 9000번 사이인 모든 요청을 방화벽 규칙 맨 위에서 최우선으로 허용해라.”

개인 서버에서 방화벽을 허용해준 다음, 공용 서버 → 개인 서버로 서버, 클라이언트 파일을 둘 다 넘겨주자.

> 공용 서버에서는 방화벽이 허용 되었다고 가정

`cp`를 이용해서 server와 client를 각각 복사시켜준다.

```bash
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5007))
server_socket.listen(1)

print("Echo Server Started!")
client_socket, addr = server_socket.accept()
print(f"Client connected: {addr}")

while True:
  data = client_socket.recv(1024)
  if not data:
    break
  received_msg = data.decode()
  print(f"Received: {received_msg}")

  msg_to_send = f"from server -> {received_msg}"

  client_socket.send(msg_to_send.encode())

client_socket.close()
server_socket.close()
```

아까는 `localhost`라고 했는데 `'0,0,0,0'` 이라고 하면 어디서든지 접속을 허용한다는 뜻이다.

개인 서버의 server 파일과 공용 서버의 server 파일 모두 `'0,0,0,0'`으로 설정해준다.

```python
import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('158.247.210.130', 5007))

while True:
  msg = input("보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.): \n")
  if msg in ["q", "quit", ""]:
    break

  client_socket.send(msg.encode())
  data = client_socket.recv(1024)
  print(f"Echo: {data.decode()}")

client_socket.close()
```

개인 서버에서 공용 서버로 접속하는 것이기 때문에 connect의 ip 부분을 공용 서버의 ip로 설정해줘야 한다.

```python
import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('161.118.158.91', 5007))

while True:
  msg = input("보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.): \n")
  if msg in ["q", "quit", ""]:
    break

  client_socket.send(msg.encode())
  data = client_socket.recv(1024)
  print(f"Echo: {data.decode()}")

client_socket.close()
```

반대로, 공용 서버에서는 개인 서버로 접속하기 위해 connect의 ip 부분을 개인 서버의 ip로 설정해줘야 한다.

## 메세지를 파일에 저장하기

개인 → 서버로 보낸 메세지들을 파일에 저장할 수 있다.

추가 기능으로 몇 시에 보냈는지 `datetime`을 이용해 보기 좋게 표시할 수 있다.

```bash
import socket
import datetime

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 5007))
server_socket.listen(5)

print("Echo Server Started! Waiting for clients...")

is_shutting_down = False

while not is_shutting_down:
    try:
        client_socket, addr = server_socket.accept()
        print(f"Client connected: {addr}")

        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    print(f"Client disconnected: {addr}")
                    break

                received_msg = data.decode()
                print(f"Received from {addr}: {received_msg}")

                if received_msg.lower() in ["q", "quit"]:
                    print("Shutdown command received. Server is shutting down.")
                    is_shutting_down = True
                    break

                now = datetime.datetime.now()
                timestamp = now.strftime("[%Y. %m. %d. %H:%M:%S]")

                log_message = f"{timestamp} {received_msg}"
                with open("socket_message_log.txt", "a", encoding="utf-8") as f:
                    f.write(log_message + "\n")

                echo_msg = f"Log saved: {received_msg}"
                client_socket.send(echo_msg.encode())

        except ConnectionResetError:
            print(f"Connection lost with {addr}")
        finally:
            client_socket.close()

    except KeyboardInterrupt:
        print("\nServer is shutting down by Ctrl+C.")
        break
    except Exception as e:
        print(f"A critical error occurred: {e}")
        continue

print("Server socket closed. Goodbye!")
server_socket.close()
```

개인 서버에서 메세지를 보낼 때마다 파일에 저장을 해야 하므로 전체를 `while`문으로 감싸준다.

`timestamp`를 현재 시각으로 설정해준 뒤, `log_message`에 `timestamp`와 `received_msg`를 합쳐서 `socket_message_log.txt`라는 파일에 적어준다.

```bash
[2025. 09. 26. 16:37:24] 안녕
[2025. 09. 26. 16:37:48] 오늘은 25년 9월 26일.
[2025. 09. 26. 16:37:59] 내일은 토요일이야
[2025. 09. 26. 16:38:09] 드디어 토요일이라니 주말이 이렇게 소중한줄 몰랐어
[2025. 09. 26. 16:38:10] 잘가
```

공용 서버의 `socket_message_log.txt`를 확인해보면 개인 서버에서 보낸 메세지들이 보낸 시각과 함께 저장되는 모습을 볼 수 있다.

```bash
ubuntu@edu007-20250924-1450:~/work/test$ python3 simple_echo_client2.py
보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.):
오늘은 25년 9월 26일.
Echo: Log saved: 오늘은 25년 9월 26일.
보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.):
내일은 토요일이야
Echo: Log saved: 내일은 토요일이야
보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.):
드디어 토요일이라니 주말이 이렇게 소중한줄 몰랐어
Echo: Log saved: 드디어 토요일이라니 주말이 이렇게 소중한줄 몰랐어
보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.):
잘가
Echo: Log saved: 잘가
보낼 메세지를 적으세요. (끝내시려면 q를 입력하세요.):
q
```
