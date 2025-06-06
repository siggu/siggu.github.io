---
layout: single
title: "모던 자바스크립트 스터디 2주차"
categories: [modern-javascript]
---

> 2025-04-09 모던 자바스크립트 스터디 2주차

> 4.6장 ~ 10.1장

# 4장

## 4.6 값의 할당

- 가비지 콜렉터
  - 변수는 하나의 값을 저장하기 위해 **확보**한 메모리 공간 그 자체
  - 여기서 확보(allocate)란, 다른 소프트웨어가 그 메모리 공간을 못 쓰게 하는 것 - 그 공간이 점점 늘어나면 사용할 수 있는 공간이 줄어들게 됨 - 이를 막기 위한 가비지 콜렉터가 존재한다.
- 언매니지드 언어, 매니지드 언어
  - 가비지 콜렉터 → 언매니지드 언어
  - 개발자가 가비지 콜렉터의 도움 없이 메모리 누수를 관리하는 것 → 매니지드 언어
    - 한 번쯤은 매니지드 언어를 공부할 가치가 있다.

## 4.7 식별자 네이밍 규칙

```js
// 카멜 케이스(camelCase)
var firstName;

// 스네이크 케이스(snake_case)
var first_name;

// 파스칼 케이스(PascalCase)
// 특수한 경우에 사용함
var FirstName;
```

# 5장 표현식과 문

> 용어가 참 중요하다. 미스 커뮤니케이션은 서로 말이 통하지 않아서 주로 발생함

## 5.1 값

: 식(표현식)이 평가되어 생성된 결과를 말한다.

## 5.4 문

> 문장을 이루는 구 라고 생각

: 프로그램을 구성하는 기본 단위이자 최소 실행 단위

### 완료값

- `var a = 1;`

  - 얘는 표현식이 아닌 문이다.

    ```bash
    // 표현식이 아닌 경우
    > var a = 1;
    < undefined

    // 표현식인 경우
    > a
    < 1
    ```

    - 표현식인 경우 **평가되는 값**을 출력한다. 그런데, 표현식이 아닌 경우 평가되는 값이 없는데 `undefined`가 출력된다.
      - 빈 값으로 놔두면 빈 문자열로 오해할 수 있으니 **완료값**으로 `undefined`를 출력해줌

## 5.2 리터럴

: **사람이 이해할 수 있는 문자 또는 약속된 기호를 사용해 값을 생성하는 표기 방식**

```
var a = b = 1;
```

- 여기에서 `a`는 변수이지만, `b`는 변수가 아니다. **window 객체의 프로퍼티**이다.
  - `delete a` -> `false`(지울 수 없음)
  - `delete b` -> `true`(지울 수 있음, 객체의 프로퍼티는 삭제 가능)

# 6장 데이터 타입

## 6.1 숫자 타입

- 자바스크립트는 하나의 숫자 타입만 존재 (실수)
  - 정수가 없느냐? 사실 정수가 있음
    - 자바스크립트 엔진이 최적화를 하고 있어서 사실은 정수가 있다. (나중에 살펴볼 예정)

## 6.9 데이터 타입의 필요성

> 데이터 타입이란, 데이터(값)에 타입이 있다는 것이다. 왜 종류가 있어야 하는가?

- `var score = 100;`
  - `score`라는 식별자가 가리키는 메모리 공간 안에 100이 있을 것이다.
  - 컴퓨터 입장에서는, 컴퓨터가 100이라는 숫자 값을 집어넣기 위한 최적의 값을 찾아야 한다.
    - 즉, 데이터 타입은 데이터를 저장하기 위해 확보할 메모리 공간의 메타 데이터이다.

## 6.10 동적 타이핑

### 정적 타입 언어

```java
char c;

int num;
```

- 변수를 선언할 때 변수에 할당할 수 있는 값의 종류, 즉 데이터 타입을 사전에 선언해야 한다.
  - 변수의 타입을 변경할 수 없는 언어를 **정적 타입 언어**라고 한다.

### 동적 타입 언어

```js
var foo;
console.log(typeof foo);  // undefined

foo = 3;
console.log(typeof foo);  // 3

foo = 'Hello';
console.log(typeof foo);  // string

...
```

- 자바스크립트의 변수에는 어떤 데이터 타입의 값이라도 자유롭게 할당할 수 있음(**동적 타입 언어**)

### 동적 타입 언어의 단점

- 데이터 타입에 무감각해질 정도로 편리하다. (-> 데이터 타입이 정해져 있지 않아서 뭐든지 들어갈 수 있다.)
  - 결국 안정성이 떨어짐 (-> 버그가 발생하기 쉽다)
  - 타입스크립트를 쓰는 이유 (-> 동적 타입 언어인 자바스크립트를 정적 타입 언어로 사용하기 위함)
    - 동적 타입 언어는 데이터 타입이 정해져 있지 않기 때문에, 변수의 경우 언제든지 변경이 되어 값을 확인하기 전까지 변수의 타입을 확인할 수 없다. 변수 안에 내가 기대하지 않은 값이 들어있는 경우가 발생할 수 있다.

# 9장 타입 변환과 단축 평가

- 개발자가 의도적으로 값의 타입을 변환하는 것을 **명시적 타입 변환, 타입 캐스팅** 이라 한다.
- 개발자의 의도와는 상관없이 표현식을 평가하는 도중에 자바스크립트 엔진에 의해 암묵적으로 타입이 자동변환되기도 한다. 이를 **암묵적 타입 변환 또는 타입 강제 변환**이라고 한다.

### 9.2.3 불리언 타입으로 변환

- `false`로 평가되는 **Falsy** 값

  - `false`
  - `undefined`
  - `null`
  - 0, -0
  - `NaN`
  - ‘’(빈 문자열)

- 위 `false`로 평가되는 **Falsy** 값을 제외한 나머지는 모두 **Truthy** 값이다.
  - 빈 배열 (`[]`) -> Truthy
  - 빈 객체 (`{}`) -> Truthy

## 9.4 단축 평가

: 계산한다는 것

- 자바스크립트에서 계산은 숫자 뿐만 아니라, 문자열, 객체, 불리언 등등을 다루기 때문에, 계산이라고 하지 않고 평가, 연산 이라고 한다.
- 표현식을 평가하면 값이 나옴.
  - 즉, 평가하여 나온 값을 단축한다는 것이다.

> 단축 평가로 대체해도 속도 차이는 **무의미**하다.

# 10장 객체

## 10.1 객체란?

> 이데아의 구체적인 정의는 현실 너머의 진짜 세상을 의미하며 지금 우리가 살고 있는 세상 즉, **현상계**는 인간의 이성에 비하면 덧없는 그림자일 뿐이고 이데아는 현상계에 가려진 뒷면인 사물의 본성이자 원형을 의미하며 죄악, 고통, 부조리가 존재하지 않는 곳으로 순수, 영원, 불변의 **이상 세계**를 말한다.

> [플라톤](https://namu.wiki/w/%ED%94%8C%EB%9D%BC%ED%86%A4)철학에 따르면 정말로 존재하는 것이고, 우리가 살아가는 세상에 존재하는 모든 것들이 이 이데아를 본뜬 것이라는 이론. 이른바 [보편자](https://namu.wiki/w/%EB%B3%B4%ED%8E%B8%EC%9E%90).플라톤에 따르면 물질적인 사물은 이데아에 비하면 그림자나 다름없다.

| 출처: 나무위키-플라톤(https://namu.wiki/w/%ED%94%8C%EB%9D%BC%ED%86%A4)

- 우리가 만든 애플리케이션을 **하나의 세상**이라고 간주해보자.
  - 그 애플리케이션 안에서 생활하고 동작하고 움직이는 무언가들의 대상이 존재하는데, 그것들이 **객체**이다.

---

# 2주차 회고

1주차에 진도를 많이 못 나가서 2주차에 빨리 진행하느라 정신이 없었다. 모든 부분을 적으려고 한다기 보다는 중요하게 설명하시는 부분을 집중적으로 필기해야 겠다고 생각했다.

10장에서 `delete` 연산자가 어떤 방식으로 동작하는지 궁금했었는데, 그 부분까지 진도는 나가지 않았지만 중간에 "`delete` 연산자가 `configurable` 여부를 확인해서 삭제할 수 있는지 없는지의 여부만 반환한다."는 것을 알게 되어 궁금증이 해결되었다. 1주차에서 책을 읽으면서 궁금증이 해소되는 부분도 좋았지만, 강의를 통해 실시간으로 궁금증이 해소되는 부분도 있어서 듣길 잘했다는 생각을 했다.

"표현식", "문", "리터럴" 이라는 표현이 아직 명확하게 이해되지 않았지만 이런 정확한 표현을 해야 미스 커뮤니케이션을 줄일 수 있을 것 같다는 생각이 들었다.

자바스크립트에서 타입스크립트가 나오고, 왜 타입스크립트를 사용해야 하는지 막연하게 타입이 존재하면 어떤 데이터 타입이 들어가는지 알 수 있다 라고만 생각하고 있었는데, 자바스크립트는 동적 타입 언어이고, 동적 타입 언어의 단점을 알고나니 타입스크립트를 도입하게 된 이유를 명확하게 알 수 있었다.
