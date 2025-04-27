---
layout: single
title: "타입스크립트의 타입과 인터페이스의 차이"
categories: [til]
subcategory: TypeScript
---

> 2025-04-17 TIL-타입스크립트의 타입과 인터페이스의 차이

# 타입스크립트의 타입과 인터페이스의 차이

타입스크립트에서 `type`과 `interface`는 둘 다 타입을 정의하는 데 쓰이지만, 쓰임새와 특징이 좀 다르다.

## interface

- 주로 **객체의 형태 정의**에 쓰인다.
- **선언 병합**이 가능해서 같은 이름으로 여러 번 선언하면 자동으로 합쳐진다.
- `extends` 키워드를 사용해서 다른 인터페이스를 **확장**할 수도 있다.
- 클래스에서 `implemnets`할 때 자연스럽게 사용된다.

### interface 예시

```ts
interface Person {
  name: string;
}

interface Person {
  age: number;
}

const p: Person = {
  name: "Alice",
  age: 20,
};
```

## type

- 객체뿐만 아니라 **기본 타입, 유니온, 인터섹션, 튜플 등 더 다양한 타입 표현**이 가능하다.
- 선언 병합이 안 된다.
- 인터페이스처럼 확장도 가능하고, 조합도 할 수 있다.

### type 예시

```ts
type A = { a: string };
type B = { b: number };

type C = A & B;

const obj: C = {
  a: "hi",
  b: 123,
};
```

## type과 interface 사용 시기

- 객체 구조를 표현하거나 클래스에 붙힐 때 `interface` 사용
- 유니온 타입, 튜플, 조건부 타입, 매핑 타입 등 **복잡한 조합이 필요할 때** `type` 사용

## 추가로 학습할 만한 자료

[[F-Lab] 타입스크립트에서 타입과 인터페이스의 차이점 이해하기](#https://f-lab.kr/insight/typescript-type-vs-interface-20240801?gad_source=1&gad_source=1&gclid=Cj0KCQiAs5i8BhDmARIsAGE4xHwNInBLKRs3N4UC8MmWPHxUB6qvBo9LhNWqlFQJ0zHBGlwgZL5j6tkaAm5oEALw_wcB)
