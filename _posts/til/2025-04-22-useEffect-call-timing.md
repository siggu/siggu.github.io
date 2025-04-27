---
layout: single
title: "useEffect 호출 시점"
categories: [til]
subcategory: React
---

> 2025-04-22TIL-useEffect 호출 시점

# useEffect 호출 시점

`useEffect`는 React에서 컴포넌트의 **생명주기(lifecycle)**에 따라 **부수 효과(side effects)**를 처리할 수 있도록 도와주는 훅이다. 이 훅은 다음 세 가지 주요 시점에서 호출된다:

## 1. 컴포넌트가 마운트될 때 (처음 렌더링 이후)

`useEffect`는 컴포넌트가 화면에 처음 렌더링된 직후 호출된다. 이 시점에서 다음과 같은 작업을 수행할 수 있다.

- 외부 API 데이터 요청
- 이벤트 리스너 등록
- 타이머 설정
- 외부 라이브러리 초기화 등

```jsx
useEffect(() => {
  console.log("컴포넌트가 마운트됨");
}, []);
```

이처럼 의존성 배열을 빈 배열(`[]`)로 전달하면 **최초 한 번만 실행**된다.

## 2. 의존성 배열 내 값이 변경될 때 (업데이트 시)

의존성 배열(`deps`)에 전달한 값 중 하나라도 변경되면 `useEffect`가 다시 호출된다.

```jsx
useEffect(() => {
  console.log("count 값이 변경됨");
}, [count]);
```

이때 흐름은 다음과 같다.

1. 먼저 **이전 effect의 클린업 함수(clean-up function)**가 호출된다.
2. 이후 새로운 effect 본문이 실행된다.

이러한 방식으로 `useEffect`는 이전 effect를 정리한 후 새로운 로직을 적용하여, 리소스 누수나 중복 동작을 방지한다.

## 3. 컴포넌트가 언마운트될 때 (제거 시)

`useEffect` 내부에서 반환한 함수는 컴포넌트가 언마운트될 때 자동으로 실행된다. 이를 **클린업(clean-up function)**라고 한다. 주로 다음 작업 때 사용된다.

1. 이벤트 리스너 제거
2. 타이머 해제
3. 외부 구독 취소 등

```jsx
useEffect(() => {
  const interval = setInterval(() => {
    console.log("실행 중...");
  , 1000);

  return () => {
    clearInterval(interval);  // 컴포넌트 언마운트 시 실행
    console.log("컴포넌트가 언마운트됨");
  };
}, []);
```

## 의존성 배열이 없는 경우

의존성 배열을 아예 전달하지 않으면, `useEffect`는 **모든 렌더링 후에 무조건 실행**된다.

```jsx
useEffect(() => {
  console.log("렌더링마다 호출됨");
});
```

## useEffect 호출 시점 요약

| 호출 시점   | 설명                                                              |
| ----------- | ----------------------------------------------------------------- |
| 마운트 후   | 의존성 배열이 빈 배열(`[]`)일 경우, 최초 1회만 실행됨             |
| 값 변경 시  | 배열 내 값 중 하나라도 변경되면, 클린업 → 본문 실행 순으로 재호출 |
| 언마운트 시 | effect가 반환한 함수가 실행되어 정리 작업 수행                    |
| 배열 없음   | 모든 렌더링 후마다 실행됨                                         |

## 추가로 학습할 만한 자료

- [[React] effect 정리가 업데이트 시마다 실행되는 이유](https://ko.legacy.reactjs.org/docs/hooks-effect.html#explanation-why-effects-run-on-each-update)
- [[React] 외부 시스템과 연결](https://ko.react.dev/reference/react/useEffect#connecting-to-an-external-system)
