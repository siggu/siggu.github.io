---
layout: single
title: "reflow와 repaint"
categories: [til]
# tags: ["머신러닝", "웹앱"]
subcategory: Web
---

> 2025-03-22 TIL-reflow와 repaint

# reflow와 repaint

## reflow

**reflow**는 <U>브라우저가 페이지의 레이아웃을 다시 계산하는 과정을 말한다.</U> DOM의 구조가 변경되거나 CSS 스타일이 변경되면, 브라우저는 각 요소가 화면에 어떻게 배치될지 다시 계산해야 한다. 이 과정은 모든 자식 요소와 관련된 부모 요소까지 영향을 주기 때문에 비용이 많이 드는 작업이다.

### reflow의 원인

- DOM 구조 변경: 새로운 노드 추가/삭제

- CSS 변경: width, height, margin, padding, border, display, position, top, left, right, bottom 등의 변경

- 브라우저 창 크기 변경: resize 이벤트 발생

- 폰트 변경: 텍스트 크기나 스타일 변경

- 스크롤: 특정 조건에서 레이아웃이 변경될 때

- CSS 속성 조회: `offsetWidth`, `offsetHeight`, `clientWidth`, `clientHeight`, `getBoundingClientRect()` 등의 속성 접근 시

## repaint

- **repaint**는 <U>요소의 모양이나 스타일이 변경될 때 발생한다.</U> 요소의 레이아웃은 그대로이고, 색상이나 배경 등의 스타일만 변경되는 경우를 말한다.

  - `background-color` 같은 속성을 예로 들 수 있다. 이 경우 브라우저는 요소의 모양만 다시 그리면 되기 때문에 `reflow`보다는 비용이 덜 들지만, 여전히 성능에 영향을 줄 수 있다.

### repaint의 원인

- 색상 변경: `color`, `background-color`

- 그림자 효과: `box-shadow`

- 투명도 변경: `opacity`

- border 스타일 변경

- visibility 변경 (`visibility: hidden;` → `display: none;`은 Reflow 발생)

> `reflow`는 *레이아웃을 다시 계산하는 과정*이고, `repaint`는 *계산 결과를 화면에 다시 그리는 과정*이라고 할 수 있다.

## reflow와 repaint 최적화 방법

1. `reflow`를 유발하는 CSS 속성 사용 최소화

   `width`, `height`, `margin`, `padding`, `bordr` 등의 속성은 요소의 레이아웃을 다시 계산하므로 reflow를 일으킨다. 가능한 한 미리 CSS에서 스타일을 설정해 초기 로드 시에만 계산이 이루어지도록 하고, 이후에는 변경하지 않는 것이 좋다.

2. CSS 애니메이션 최적화

   애니메이션에 `transform`과 `opacity` 속성만을 사용하는 것이 성능에 유리하다. 이 두 속성은 GPU 가속을 사용할 수 있어 reflow를 일으키지 않고 repaint만 발생시키므로 CPU 자원을 적게 사용한다.

3. `will-change` 속성 사용

   CSS의 `will-change` 속성을 사용하여 브라우저에 특정 요소가 변경될 것이라고 미리 언질을 줄 수 있다.

   ```js
   .animated-element {
       will-change: transform, opacity;
   }
   ```

   하지만 `will-change` 속성은 너무 자주 사용하면 메모리 낭비가 발생하므로 필요한 요소에만 적용해야 한다.
