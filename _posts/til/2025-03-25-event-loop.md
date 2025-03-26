---
layout: single
title: "이벤트 루프"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-03-25 TIL-이벤트 루프

# 이벤트 루프

자바스크립트의 **이벤트 루프 (Event Loop)**는 자바스크립트가 싱글 스레드 기반 언어임에도 불구하고 <U>비동기 작업을 처리할 수 있게 해주는 메커니즘</U>이다.

자바스크립트는 기본적으로 한 번에 하나의 작업만 처리할 수 있다. 하지만 이벤트 루프가 **콜 스택**과 **태스크 큐**를 관리하면서 비동기 작업이 완료되면 그 결과를 처리할 수 있게 도와준다.

> **콜 스택**은 현재 실행중인 코드들이 쌓이는 곳이고, **태스트 큐**는 비동기 작업이 완료되면 그 결과를 대기시키는 곳이다.

## 이벤트 루프의 동작 방식

1. 콜 스택 (Call Stack)

   - 자바스크립트 엔진이 실행할 코드 (함수)를 쌓아두는 곳이다.

   - 함수가 호출되면 스택에 쌓이고, 실행이 끝나면 제거된다.

2. 웹 API (또는 Node.js API)

   - `setTimeout`, `fetch`, `addEventListener` 같은 비동기 작업을 처리하는 브라우저 (Node.js) 제공 API이다.

   - 비동기 작업이 실행되면, 해당 API가 백그라운드에서 처리한다.

3. 태스크 큐 (Task Queue) / 마이크로태스크 큐 (Microtask Queue)

   - 비동기 작업이 끝나면 해당 콜백 함수가 큐에 들어가고, 이벤트 루프가 이를 콜 스택으로 이동시켜 실행한다.

     > 마이크로태스크 큐: `Promise.then`, `MutationObserver` 등의 콜백이 저장되는 우선순위가 높은 큐

     > 태스크 큐: `setTimeout`, `setInterval`, `I/O 작업` 같은 일반 비동기 작업이 저장되는 큐

4. 이벤트 루프 (Event Loop)

   - 이벤트 루프는 계속해서 "콜 스택이 비었는지" 확인한다.

   - 비어있다면, 태스크 큐 또는 마이크로태스크 큐에서 콜백을 가져와 실행한다.

## 실행 흐름 예제

```js
console.log("Start");

setTimeout(() => {
  console.log("setTimeout");
}, 0);

Promise.resolve().then(() => {
  console.log("Promise");
});

console.log("End");
```

✅ **실행 순서**

1. `"Start"`가 **콜 스택**에서 실행됨 -> 출력

2. `setTimeout`이 **Web API**로 전달됨 (콜백을 태스크 큐에 저장)

3. `Promise.resolve().then(...)`이 **마이크로태스크 큐**에 저장됨

4. `"End"`가 **콜 스택**에서 실행됨 -> 출력

5. 이벤트 루프가 **마이크로태스크 큐**를 먼저 실행 -> `"Promise"` 출력

6. 이벤트 루프가 **태스크 큐**에서 `setTimeout` 콜백을 실행 -> `"setTimeout"` 출력

**최종 출력 결과**

```
Start
End
Promise
setTimeout
```
