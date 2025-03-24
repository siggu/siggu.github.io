---
layout: single
title: "실행 컨텍스트"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-03-23 TIL-실행 컨텍스트

# 실행 컨텍스트

**실행 컨텍스트**는 자바 스크립트에서 코드가 실행되는 환경을 의미한다. 자바스크립트 엔진이 코드를 실행할 때, 그 코드가 실행될 때의 환경을 정의하고 관리하기 위해 존재하는 것이 실행 컨텍스트이다.

## 실행 컨텍스트의 종류

1. 전역 실행 컨텍스트 (Global Execution Context)

   - 자바 스크립트 코드를 실행하면 가장 먼저 생성됨.

   - 전역 코드 (최상휘 스코프의 코드)를 실행하는 컨텍스트

   - `this`가 전역 객체(window 또는 globalThis)를 가리킴.

2. 함수 실행 컨텍스트 (Function Execution Context)

   - 함수가 호출될 때마다 생성됨.

   - 함수의 실행을 위한 환경을 제공하며, 함수 실행이 끝나면 제거됨.

3. Eval 실행 컨텍스트 (Eval Execution Context)

   - `eval()` 함수가 실행될 때 생성됨.

   - 보안 및 성능 문제 때문에 거의 사용되지 않음.

## 실행 컨텍스트의 생성 과정

1. **생성 단계 (Creation Phase)**

   - `this` 바인딩 결정

   - Lexical Environment(렉시컬 환경) 생성

     - 환경 레코드(Environment Record): 변수, 함수 선언 저장

     - 외부 렉시컬 환경 참조 (Outer Lexical Environment Reference): 스코프 체인 형성

   - Variable Environment (변수 환경) 생성

     - `var` 변수는 `undefined`로 초기화됨 (호이스팅 발생)

2. **실행 단계 (Execution Phase)**

   - 코드가 한 줄씩 실행되면서 변수에 값이 할당됨

## 실행 컨텍스트의 구성 요소

1. **Lexical Environment (렉시컬 환경)**

   - 변수와 함수 선언을 저장하는 공간

   - `let`, `const` 변수는 TDZ(Temporal Dead Zone)로 인해 선언 전에 참조 불가

2. **스코프 체인 (Scope Chain)**

   - 실행 컨텍스트가 중첩될 때, 내부에서 선언한 변수를 찾지 못하면 **외부 컨텍스트를 탐색**하는 방식

   - 내부 함수가 외부 함수의 변수를 참조할 수 있도록 함

3. **this 바인딩**

   - 전역 실행 컨텍스트: `this`는 `window` (브라우저) 또는 `globalThis` (Node.js)

   - 함수 실행 컨텍스트: 함수 호출 방식에 따라 `this`가 결정됨

   - 화살표 함수는 `this`를 바인딩하지 않고 상위 스코프의 `this`를 유지함.

## 실행 컨텍스트와 콜 스택

```js
function a() {
  console.log("a");
}
function b() {
  a();
  console.log("b");
}
function c() {
  b();
  console.log("c");
}
c();
```

**실행 과정**

1. 전역 실행 컨텍스트 (GEC) 생성 -> 콜 스택에 push

2. `c()` 호출 -> `c` 실행 컨텍스트 push

3. `b()` 호출 -> `b` 실행 컨텍스트 push

4. `a()` 호출 -> `a` 실행 컨텍스트 push

5. `a()` 실행 완료 -> 실행 컨텍스트 pop

6. `b()` 실행 완료 -> 실행 컨텍스트 pop

7. `c()` 실행 완료 -> 실행 컨텍스트 pop

8. 전역 실행 컨텍스트 유지

**콜 스택 (Stack)**

1. GEC (전역 실행 컨텍스트)

2. third 실행 컨텍스트

3. second 실행 컨텍스트

4. first 실행 컨텍스트 (최상단 → 실행 후 pop)

## 실행 컨텍스트와 `this`

실행 컨텍스트에 따라 `this`가 달라진다.

```js
console.log(this); // window (브라우저)

function regularFunction() {
  console.log(this);
}
regularFunction(); // window (전역 호출)

const obj = {
  method: function () {
    console.log(this);
  },
};
obj.method(); // obj (메서드 호출)

const arrowFunction = () => {
  console.log(this);
};
arrowFunction(); // window (화살표 함수는 this를 바인딩하지 않음)
```

**화살표 함수와 일반 함수의 차이**

- 일반 함수: `this`는 호출한 객체에 바인딩됨.

- 화살표 함수: `this`는 자신이 선언된 **스코프의 this**를 유지.
