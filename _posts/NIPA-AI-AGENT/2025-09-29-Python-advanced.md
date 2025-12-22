---
layout: single
title: "파이썬 고급 - 캡슐화, 예외처리, 라이브러리"
categories: [nipa-ai-agent]
subcategory: Python
---

> 2025-09-29 nipa-ai-agent-파이썬-고급

## 캡슐화와 getter/setter

```python
# 캡슐화와 getter/setter
class BankAccount:
  use_count = 0

  def __init__(self, owner, balance):
    self.owner = owner
    self.__balance = balance # private 변수

  def deposit(self, amount):
    self.__balance += amount
    print(f"{self.owner}님의 계좌에 {amount}원을 추가했습니다.\n")

  def withdraw(self, amount):
    if amount <= self.__balance:
      self.__balance -= amount
      print(f"{self.owner}님의 계좌에서  {amount}만큼 인출하여 {self.__balance}만큼 남았습니다.")
    else:
      BankAccount.use_count -= 1
      print(f"{self.owner}님의  계좌에  {amount - self.__balance}원이 부족합니다.\n")

  def get_balance(self):
    return self.__balance

# 사용
suzi_acc = BankAccount("수지", int(input(f"현재 계좌가 얼마인가요?\n")))

while True:
  user_input = input("입금하시려면 '입금', 출금하시려면 '출금'이라고 적어주세요. 그만하시려면 '나가기'를 입력해주세요. \n")

  if user_input == "입금":
    BankAccount.use_count += 1
    print(f"{suzi_acc.owner}님의 현재  잔액:\n", suzi_acc.get_balance())
    suzi_acc.deposit(int(input("입금할 금액을 입력해주세요.\n")))
  elif user_input == "출금":
    BankAccount.use_count += 1
    print(f"{suzi_acc.owner}님의 현재 잔액은 {suzi_acc.get_balance()}원 입니다.")
    suzi_acc.withdraw(int(input("출금할 금액을 입력해주세요.\n")))
  elif user_input == "나가기":
    break
  else:
    print("올바른 단어를 입력해주세요.")

  print(f"{suzi_acc.owner}님의 현재 잔액은 {suzi_acc.get_balance()}원 입니다.")

print(f"\n{suzi_acc.owner}님의 현재 잔액은 {suzi_acc.get_balance()}원입니다. 이용해주셔서 감사합니다.")
```

## 예외처리

**ERROR_COMMENT.json**

```json
{
  "ERROR": {
    "VALUE_ERROR": "숫자가 아닙니다.",
    "ZERO_DIVISION_ERROR": "0으로 나눌 수 없습니다."
  }
}
```

**예외처리 코드**

```python
import json

# 데이터를 저장할 변수 초기화
data = None

try:
    # 'with'를 사용해 파일을 열면 코드가 끝나거나 에러가 발생했을 때
    # 자동으로 파일을 닫아주어 편리합니다.
    # 'utf-8' 인코딩은 한글이 깨지는 것을 방지합니다.
    with open('ERROR_COMMENT.json', 'r', encoding='utf-8') as file:
        # json.load() 함수를 사용해 JSON 파일을 파이썬 딕셔너리로 불러옵니다.
        data = json.load(file)

except FileNotFoundError:
    print("오류: 'ERROR_COMMENT.json' 파일을 찾을 수 없습니다.")
except json.JSONDecodeError:
    print("오류: 파일이 올바른 JSON 형식이 아닙니다.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")

# 파일에서 불러온 데이터를 성공적으로 사용
if data:
  try:
    num = int(input("10을 입력한 숫자로 나누는 연산입니다. 숫자를 입력해주세요. \n"))
    result = int(10 / num)
    print(f"result: {result}")
  except ValueError:
    print(data['ERROR']['VALUE_ERROR'])
  except ZeroDivisionError:
    print(data['ERROR']['ZERO_DIVISION_ERROR'])
  finally:
    print("연산이 완료되었습니다.")
```

## 라이브러리

```python
import random

dice = random.randint(1, 6)
print("랜덤 주사위 수: ", dice)

import datetime

today = datetime.date.today()
print("오늘 날짜: ",today)

import time

print(time.strftime("%Y.%m.%d~%H:%M:%S"))
```
