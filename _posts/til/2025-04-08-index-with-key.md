---
layout: single
title: "리액트에서 index를 key값으로 사용하면 안 되는 이유"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-04-07 TIL-리액트에서 index를 key값으로 사용하면 안 되는 이유

# 리액트에서 index를 key값으로 사용하면 안 되는 이유

- 리액트에서 `index`를 `key`로 사용하는 것은 권장되지 않는다.

- 그 이유는 **배열의 요소들이 추가되거나 삭제될 때, 배열의 순서가 바뀌는 경우** 문제가 발생할 수 있기 때문이다.

## key의 역할

리액트에서 `key`는 컴포넌트의 **고유성 식별자**이다. 리스트를 렌더링할 때, 어떤 항목이 **추가/삭제/변경**되었는지 리액트가 빠르게 파악해서 **최소한의 DOM 변경**을 하기 위해 쓴다.

## 문제점

- `index`를 `key`로 사용하면, 배열의 순서가 변경될 때 리액트가 요소들을 잘못 인식할 수 있다. 예를 들어, 배열에 새로운 요소가 추가되면 그 뒤에 있는 요소들의 인덱스가 모두 바뀌게 된다. 리액트는 이를 새로운 요소로 인식해 불필요하게 재렌더링 하거나, 요소의 상태를 잘못 처리할 수 있다.

- 이로 인해 성능 문제가 발생하거나, 사용자 입력 상태 같은 요소가 의도치 않게 초기화되는 등 예기치 않은 버그가 생길 수 있다. 그래서 배열의 순서나 요소 변경에 영향을 받지 않는 고유한 값을 `key`로 사용하는 것이 좋다.

## key로 사용할 고유 값을 생성하는 방법

- 주로 데이터의 유일성을 보장하고 변하지 않는 값을 사용하는 것이 중요하다.

- 서버의 데이터베이스에서 제공하는 **고유 ID**를 사용하는 것이 가장 권장된다. 만약 이 방법이 불가능할 경우, `${item.title}_${item.username}`와 같은 형태로 여러 필드를 결합하여 고유 값을 생성할 수 있다. 혹은, 렌더링 이전 시점에 UUID 혹은 랜덤 값을 생성하여 고유 값을 부여할 수 있다.

## 추가로 학습할 만한 자료

- [[Robin Porkorny]Index as a key is an anti-pattern](https://robinpokorny.medium.com/index-as-a-key-is-an-anti-pattern-e0349aece318)
- [[Jeongkuk Seo] 배열의 index를 key로 쓰면 안되는 이유](https://medium.com/sjk5766/react-%EB%B0%B0%EC%97%B4%EC%9D%98-index%EB%A5%BC-key%EB%A1%9C-%EC%93%B0%EB%A9%B4-%EC%95%88%EB%90%98%EB%8A%94-%EC%9D%B4%EC%9C%A0-3ce48b3a18fb)
- [[Stack Overflow] How to create unique keys for React elements?](https://stackoverflow.com/questions/39549424/how-to-create-unique-keys-for-react-elements)
