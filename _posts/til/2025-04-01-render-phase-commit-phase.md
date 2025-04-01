---
layout: single
title: "리액트의 render pahse와 commit phase란?"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-04-01 TIL-리액트의 render pahse와 commit phase란?

# 리액트의 render pahse와 commit phase란?

리액트의 렌더링 과정은 크게 두 가지 단계로 나눌 수 있다. **render phase**와 **commit phase**이다.

각각의 단계에서 수행되는 작업과 특징을 이해하면 리액트의 렌더링 최적화와 성능 개선에 도움이 된다.

## render phase

리액트가 **변경된 상태나 props에 따라 어떤 UI가 변경되어야 할지를 결정하는 단계**이다.

### render phase의 주요 특징

- **가상 DOM(Virtual DOM) 생성 및 변경 사항 계산**

  - 컴포넌트 함수가 실행되면서 새로운 가상 DOM을 생성한다.

  - 이전 가상 DOM과 비교하여 변경된 부분을 찾는다 (Diffing 과정)

- **실제 DOM 변경 없음**

  - 이 단계에서는 실제로 DOM을 수정하지 않는다.

- **순수 함수처럼 동작해야 함**

  - `useState`, `useReducer`를 호출할 수 있지만, 사이드 이펙트(side effects)를 발생시키면 안 된다.

  - 따라서 `useEffect`, `useLayoutEffect`는 이 단계에서 실행되지 않는다.

- **Concurrent Mode 지원**

  - React 18부터 도입된 Concurrent Mode에서는 렌더링이 중단되거나 다시 실행될 수 있다.

  - 우선순위가 높은 작업이 있다면, 낮은 우선순위의 렌더링을 잠시 멈추고 먼저 실행할 수도 있다.

### render phase에서 실행되는 주요 작업

- 함수 컴포넌트 실행

- 새로운 가상 DOM 생성

- 이전 가상 DOM과 비교(Diffing)

- 변경된 부분을 찾고 업데이트할 준비

## commit phase

**render phase에서 계산된 변경 사항을 실제 DOM에 적용하는 단계**이다.

### commit phase의 주요 특징

- **실제 DOM 업데이트**

  - 변경된 가상 DOM을 기반으로, 브라우저의 실제 DOM을 수정한다.

- **사이드 이펙트 실행 가능**

  - `useEffect`, `useLayoutEffect` 같은 훅이 실행되는 시점이다.

  - 이벤트 리스너 등록, API 호출, 애니메이션 시작 등의 작업이 가능하다.

- **ref 값 업데이트 가능**

  - `useRef`로 참조하는 DOM 요소의 값이 변경된다.

### commit phase에서 실행되는 주요 작업

- 변경된 내용을 실제 DOM에 반영

- useEffect 및 useLayoutEffect 실행

- `componentDidMount`, `componentDidUpdate`, `componentWillUnmount` 실행 (클래스 컴포넌트 기준)

> 요약하면, **render phase**는 변화된 UI를 결정하는 계산 과정이고, **commit phase**는 그 계산된 결과를 실제로 반영하는 단계이다.

## render phase와 commit phase가 동기화될 때의 특징

크게 두 가지로 말할 수 있다. **단계적 진행**과 **병목 관리**이다.

### 단계적 진행

- **render phase**가 완료되면 리액트는 즉시 **commit phase**를 실행하지 않고, 다른 높은 우선순위 작업이 있다면 먼저 처리한 후 나중에 **commit phase**를 실행할 수 있다.

  - 예를 들어, `Suspense`를 활요하면 일부 데이터가 로드될 때까지 commit phase를 지연시킬 수 있다.

  - 이러한 단계적 진행을 통해 리액트는 동기화가 필요한 작업을 효율적으로 관리하여 사용자 경험을 개선한다.

### 병목 관리

- **render phase**에서 모든 변경 사항이 **Fiber Tree**에 준비된 상태에서 **commit phase**로 넘어가므로, **render**와 **commit** 단계의 일관성이 유지된다.

  - 이렇게 두 단계는 순차적으로 작동하여, UI가 정확하게 동기화되고 불필요한 재렌더링을 방지한다.
