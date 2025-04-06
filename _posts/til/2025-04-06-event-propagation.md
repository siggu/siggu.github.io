---
layout: single
title: "이벤트 전파"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-04-05 TIL-이벤트 전파

# 이벤트 전파

이벤트 전파는 DOM 요소에 이벤트가 발생했을 때, 그 이벤트가 전파되는 방식을 말한다.

이벤트 전파는 크게 세 가지 단계로 나뉜다.

## 1. 캡처링 단계 (Capturing Phase)

- 이벤트가 **최상위 요소(document 또는 window)**부터 시작해서 이벤트가 실제로 발생한 **타깃 요소까지 내려가는 과정**이다.

- 이 과정에서 상위 요소들에 이벤트 리스너가 있으면 그 순서대로 실행될 수 있다.

## 2. 타깃 단계 (Target Phase)

- 이벤트가 **목표한 요소에 도달**했을 때 실행되는 단계이다.

- 타겟 요소에 등록된 이벤트 리스너가 이 시점에 실행된다.

## 3. 버블링 단계 (Bubbling Phase)

- 이벤트가 발생한 후, **다시 DOM 트리의 상위 요소들로 이벤트가 전파되어 올라가는 단계**이다.

- 이 과정에서 상위 요소들에 등록된 이벤트 리스너들이 실행될 수 있다.

> 기본적으로 대부분의 이벤트는 버블링을 통해 전파되지만, `addEventListener`의 세 번째 인자로 `{ capture : true }`를 전달하면 캡처링 단계에서도 이벤트를 처리할 수 있다.

## 예시 코드

```html
<div id="parent">
  <button id="child">Click me</button>
</div>

<script>
  document.getElementById("parent").addEventListener("click", () => {
    console.log("Parent clicked");
  });

  document.getElementById("child").addEventListener("click", () => {
    console.log("Child clicked");
  });
</script>
```

이 상태에서 버튼을 클릭하면 콘솔에는 다음과 같이 출력된다.

```nginx
Child clicked
Parent clicked
```

> **버블링 단계** 때문. 이벤트가 `child`에서 발생하고 나서 `parent`로 전파된 것

## 이벤트 전파 막기

- `event.stopPropagation()`

  - 이벤트가 **더 이상 전파되지 않도록 중단**

- `event.stopImmediatePropagation()`

  - 전파 중단 + **같은 요소 내의 다른 리스너 호출도 막음**
