---
layout: single
title: "리액트에서 성능 최적화를 위한 방법들"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-03-28 TIL-리액트에서 성능 최적화를 위한 방법들

# 리액트에서 성능 최적화를 위한 방법들

## 메모이제이션

리액트에서 성능 최적화를 위해 여러 가지 방법을 사용할 수 있다. 대표적으로 **메모이제이션**이 있다.

### `memo`

- 리액트의 `memo`를 사용하면 컴포넌트를 메모이제이션할 수 있다.

- 이는 **컴포넌트의 props가 변경되지 않았을 때 리렌더링을 방지**하여 성능을 최적화한다.

> 특히 **렌더링 비용이 큰 컴포넌트에서 유용**하다.

### `useCallback`과 `useMemo`

- `useCallback`은 함수를 메모이제이션하여 **불필요한 함수 재생성을 방지**한다.

- `useMemo`는 **연산 비용이 큰 값을 캐싱**하여 불필요한 연산을 줄인다.

> 이를 통해 **자식 컴포넌트로 전달되는 함수나 값이 변경되지 않으면 리렌더링을 피할 수 있다.**

## 코드 스플리팅 (Code Splitting)

코드 스플리팅을 사용하면 애플리케이션을 여러 개의 작은 청크로 나누어, **필요한 코드만 로드**하도록 할 수 있다.

> 이를 통해 **초기 로딩 속도를 개선**하고 **불필요한 코드 실행을 방지**할 수 있다.

### 언제 코드 스플리팅을 사용할까?

1. **초기 로딩 시간이 길어지는 경우**

   - 애플리케이션이 커지면 모든 코드를 한 번에 로드하는 것이 비효율적이다.

   - 코드 스플리팅을 사용하면 초기 로드 시 **핵심 코드만 불러오고, 나머지는 필요할 때 로드**하여 성능을 최적화할 수 있다.

2. **라우트별 코드 분할이 필요한 경우**

   - SPA에서는 페이지마다 필요한 코드가 다르므로, **라우트별로 코드 스플리팅**을 하면 성능이 개선된다.

   - `React.lazy`와 `Suspense`를 사용하면 특정 컴포넌트를 필요할 때만 로드할 수 있다.

### 코드 스플리팅 예제 (라우트 기반)

```jsx
import React, { lazy, Suspense } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

const Home = lazy(() => import("./Home"));
const About = lazy(() => import("./About"));

function App() {
  return (
    <Router>
      <Suspense fallback={<div>Loading...</div>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Suspense>
    </Router>
  );
}

export default App;
```

- `React.lazy()`를 사용하면 **Home, About 컴포넌트가 필요할 때만 로드**됨

- `Susepense`는 컴포넌트가 로드될 때 보여줄 `fallback` UI를 제공
