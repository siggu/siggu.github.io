---
layout: single
title: "useEffect와 useLayoutEffect"
categories: [til]
# tags: ["머신러닝", "웹앱"]
subcategory: React
---

> 2025-04-09 TIL-useEffect와 useLayoutEffect

# useEffect와 useLayoutEffect

`useEffect`와 `useLayoutEffect`는 모두 렌더링된 후에 특정 작업을 수행하기 위해 사용된다. 하지만 실행되는 **타이밍**과 **용도**가 다르다.

## useEffect

- `useEffect`는 리액트가 DOM을 업데이트하고 브라우저가 화면을 그린 뒤(Painted 이후) 실행된다.

  - 즉, 화면이 실제로 사용자에게 보인 후에 **비동기적으로 실행**된다.

- 그래서 `useEffect`는 보통 데이터를 가져오는 작업이나 이벤트 리스너 추가 등 렌더링 후에 화면에 직접적인 영향을 주지 않는 작업에 주로 사용된다.

### useEffect의 주요 용도

- 데이터 요청(`fetch`)

- 이벤트 리스너 등록

- 로깅 또는 cleanup 작업

- 외부 라이브러리 연동

### useEffect 예시

```jsx
// useEffect
useEffect(() => {
  fetchData().then((data) => setData(data));
}, []);
```

## useLayoutEffect

- `useLayoutEffect`는 **렌더링 후 DOM이 업데이트되기 직전의 시점**에 **동기적**으로 실행된다.

  - **동기적**이라는 것은 화면에 내용이 그려지기 전에 모든 레이아웃 관련 작업이 완료된다는 의미이다.

- 예를 들어, **DOM의 크기를 측정**하거나 **위치를 조정해야 할** 때 `useLayoutEffect`를 사용하면 즉각적으로 그 변경사항이 반영되어 화면 깜빡임이나 불필요한 재랜더링을 방지할 수 있다.

### useLayoutEffect의 주요 용도

- DOM 크기나 위치 측정

  - 해당 측정값을 기반으로 레이아웃 조정

- 화면 렌더링에 영향을 줄 수 있는 작업

### useLayoutEffect 예시

```jsx
// useLayoutEffect
useLayoutEffect(() => {
  const height = ref.current.offsetHeight;
  if (height > 500) {
    ref.current.style.height = "500px"; // 레이아웃 조정
  }
}, []);
```

> 정리하면, **렌더링 후 실행되는 비동기 작업**에는 `useEffect`가 적합하고, 레이아웃이나 DOM 조작과 같이 **화면이 그려지기 전에 완료되어야 하는 작업**에는 `useLayoutEffect`가 적합하다.

## 주의사항

`useLayoutEffect` 사용 시 **성능 면에서 주의할 점**이 있다.

- `useLayoutEffect`는 동기적으로 실행되기 때문에 너무 많은 작업이 실행되면 렌더링이 느려질 수 있다.

  - 따라서 보통은 `useEffect`를 기본적으로 사용하고, 화면에 영향을 주는 작업만 `useLayoutEffect`로 처리하는 것이 좋다.

## 추가로 학습할 만한 자료

[[F-Lab] React의 useEffect와 useLayoutEffect 이해하기](https://f-lab.kr/insight/understanding-useeffect-and-uselayouteffect-in-react-20240618?gad_source=1&gclid=Cj0KCQiAs5i8BhDmARIsAGE4xHxS3kYk36u3puO4sBg1VUM0jPwZPrMBYk7CUGUq3wzBdt6Md0nOxRcaAgVjEALw_wcB)
