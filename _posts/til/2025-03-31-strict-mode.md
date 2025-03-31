---
layout: single
title: "리액트의 Strict Mode란?"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-03-29 TIL-리액트의 Strict Mode란?

# 리액트의 Strict Mode란?

리액트에서 `StrictMode`는 주로 개발 중에 발생할 수 있는 잠재적인 문제를 사전에 감지하고 예방하기 위해 사용된다.

## Strict Mode의 주요 기능

### 1. 오래된 라이프사이클 메서드와 비권장 API 사용 감지

- `componentWillMount`, `componentWillReceiveProps`와 같은 메서드는 더 이상 사용이 권장되지 않는데, StrictMode는 이러한 메서드들이 코드에 포함된 경우 경고를 표시해준다. 이를 통해 개발자가 최신 React API를 사용하여 보다 안정적이고 효율적인 코드를 작성하도록 돕는다.

### 2. 의도치 않은 부수 효과를 방지

- 리액트는 컴포넌트의 렌더링이 예측 가능하고 순수하게 이루어지기를 기대한다.

- StrictMode는 이를 검증하기 위해 useEffect, useState 등 일부 훅이나 메서드를 두 번씩 실행한다.

  - 이렇게 두 번 실행하는 이유는, 동일한 결과가 나오는지 확인함으로써 컴포넌트가 사이드 이펙트를 일으키지 않고 순수하게 동작하는 지를 검사하기 위함이다.

> 다만, 이러한 두 번 실행되는 현상은 개발 모드에서만 발생하고, 실제 프로덕션 빌드에서는 정상적으로 한 번만 실행되기 때문에 성능에 영향을 미치지 않는다.

### 3. 레거시 문자열 ref 감지

- `<input ref="myRef" />` 같은 문자열 ref 사용을 감지하고 경고한다.

- 대신 React.createRef() 또는 useRef()를 사용하도록 권장한다.
