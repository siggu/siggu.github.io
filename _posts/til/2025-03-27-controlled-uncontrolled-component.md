---
layout: single
title: "리액트의 Controlled Component와 Uncontrolled Component의 차이점"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-03-27 TIL-리액트의 Controlled Component와 Uncontrolled Component의 차이점

# 리액트의 Controlled Component와 Uncontrolled Component의 차이점

리액트에서 폼 요소를 다룰 때, **Controlled Component (제어 컴포넌트)**와 **Uncontrolled Component (비제어 컴포넌트)**의 개념이 있다.

## Controlled Component (제어 컴포넌트)

- **React 상태 (state)가 입력 값을 관리하는 방식**

- 입력 필드의 값이 `useState` 등의 상태값에 의해 제어됨

- 값이 변경될 때마다 `onChnage` 이벤트를 통해 상태를 업데이트

### Controlled Component 예시

```jsx
import { useState } from "react";

function ControlledInput() {
  const [text, setText] = useState("");

  return (
    <input type="text" value={text} onChange={(e) => setText(e.target.value)} />
  );
}
```

- 장점: React가 상태를 관리하므로 값 추적이 쉽고, 입력값을 검증하거나 동적으로 변경하기 용이함

- 단점: 모든 입력 변화마다 `re-render`가 발생하여 성능이 저하될 수 있음.

## Uncontrolled Component (비제어 컴포넌트)

- **DOM 자체에서 값을 관리하는 방식**

- `useRef`를 사용해 직접 DOM에서 값을 가져옴

- 입력값을 변경해도 컴포넌트 상태를 업데이트하지 않음

### Uncontrolled Component 예시

```jsx
import { useRef } from "react";

function UncontrolledInput() {
  const inputRef = useRef(null);

  const handleClick = () => {
    alert(`입력값: ${inputRef.current.value}`);
  };

  return (
    <>
      <input type="text" ref={inputRef} />
      <button onClick={handleClick}>값 확인</button>
    </>
  );
}
```

- 장점: 상태 업데이트가 없으므로 성능이 더 좋음

- 단점: Recat의 상태 관리와 동기화되지 않아 값 추적이 어려움

## Controlled Component vs Uncontrolled Component

- Controlled Component

  - 입력값을 검증해야 하거나, 입력값을 React 상태로 관리해야 할 때 사용

  - 예: 로그인 폼, 검색 입력창

- Uncontrolled Component

  - 초기 렌더링 후 값 변경이 필요하지 않거나, 성능 최적화가 필요할 때 사용

  - 예: 파일 업로드 (`<input type="file" />`)
