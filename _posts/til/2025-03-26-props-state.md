---
layout: single
title: "리액트의 props와 state"
categories: [til]
# tags: ["머신러닝", "웹앱"]
subcategory: React
---

> 2025-03-26 TIL-리액트의 props와 state

# 리액트의 props와 state

리액트에서 `props`와 `state`는 컴포넌트에서 데이터를 관리하는 두 가지 핵심 개념이다.

## Props (Properties)

- `props`는 부모 컴포넌트에서 자식 컴포넌트로 전달하는 **읽기 전용(immutable)** 데이터이다.

- 자식 컴포넌트는 `props`를 직접 변경할 수 없다.

- 부모 컴포넌트가 새로운 `props`를 전달하면, 자식 컴포넌트가 리렌더링된다.

### props 예제

```jsx
function ChildComponent({ name }) {
  return <h1>{name}</h1>;
}

function ParentComponent() {
  return <ChildComponent name="React" />;
}
```

- `ParentComponent`가 `ChildComponent`에 `name`이라는 `props`를 전달하고, 자식 컴포넌트는 이를 화면에 출력한다.

### props를 변경하면 오류 발생

```jsx
function ChildComponent(props) {
  props.name = "New Name"; // 오류 발생 가능
  return <h1>{props.name}</h1>;
}
```

- `props`는 부모 컴포넌트에서 관리하는 불변 데이터이므로, 자식 컴포넌트에서 직접 변경할 수 없다.

## State

- `state`는 컴포넌트 내부에서 관리되는 데이터이다.

- `state`는 동적으로 변경될 수 있으며, **변경되면 컴포넌트가 다시 렌더링**된다.

- `state`는 주로 **사용자 입력, 네트워크 요청 결과, UI 상태** 등을 관리할 때 사용된다.

### state 예제

```jsx
import { useState } from "react";

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>현재 카운트: {count}</p>
      <button onClick={() => setCount(count + 1)}>+1 증가</button>
    </div>
  );
}
```

- `useState(0)`을 사용해 `count` 상태를 정의하고, 버튼 클릭 시 `setCount(count + 1)`을 호출하면 `state`가 변경되어 컴포넌트가 리렌더링된다.

## Props가 자식 컴포넌트에서 변하지 않는 이유는?

`props`가 자식 컴포넌트에서 변하지 않는 이유는, **리액트의 단방향 데이터 흐름 (one-way data flow) 원칙** 때문이다.

### 단방향 데이터 흐름이란?

- 데이터는 **부모 -> 자식** 방향으로만 전달된다.

- 부모가 상태를 관리하고, 필요한 데이터를 자식에게 `props`로 전달한다.

- 자식 컴포넌트에서 `props`를 직접 수정할 수 없으며, 부모 컴포넌트가 `props`를 변경해야만 새로운 값이 전달된다.

**장점**

- 데이터 흐름을 예측할 수 있어 유지보수가 쉬움

- 불필요한 상태 변경을 방지하여 성능 최적화 가능

- 컴포넌트의 재사용성이 높아짐

## 자식 컴포넌트에서 props를 변경해야 하는 경우

자식 컴포넌트에서 `props`를 변경해야 하는 경우, **상태를 부모로 올리는 (state lifting) 기법**을 사용한다.

### 콜백 함수를 이용한 상태 끌어올리기

```jsx
function ParentComponent() {
  const [name, setName] = useState("React");

  return <ChildComponent name={name} onChangeName={setName} />;
}

function ChildComponent({ name, onChangeName }) {
  return (
    <div>
      <p>{name}</p>
      <button onClick={() => onChangeName("New Name")}>이름 변경</button>
    </div>
  );
}
```

- `onChangeName`이라는 함수를 `props`로 전달하고, 자식 컴포넌트에서 이를 호출하여 부모의 상태를 변경하도록 한다.

- 이렇게 하면 단방향 데이터 흐름을 유지하면서도, 자식 컴포넌트가 데이터를 변경할 수 있다.

## Props와 State 차이점

| 구분           | Props                                     | State                                |
| -------------- | ----------------------------------------- | ------------------------------------ |
| 데이터 소유    | 부모 컴포넌트가 관리                      | 컴포넌트 자체가 관리                 |
| 변경 가능 여부 | 읽기 전용 (immutable)                     | 변경 가능 (mutable)                  |
| 리렌더링 여부  | 부모가 새로운 `props`를 전달하면 리렌더링 | 상태가 변경되면 자동으로 리렌더링    |
| 사용 목적      | 부모 -> 자식으로 데이터 전달              | 컴포넌트 내부에서 동적인 데이터 관리 |

## Props와 State 함께 사용하기

부모 컴포넌트의 `state`를 `props`로 자식에게 전달할 수도 있다.

### props와 state 함께 사용하기 예제

```jsx
function ParentComponent() {
  const [text, setText] = useState("Hello");

  return (
    <div>
      <ChildComponent message={text} />
      <button onClick={() => setText("Hello, React!")}>Change Text</button>
    </div>
  );
}

function ChildComponent({ message }) {
  return <h1>{message}</h1>;
}
```

- 부모의 `state`가 변경되면 새로운 `props`가 전달되고, 자식 컴포넌트가 자동으로 리렌더링 된다.
