---
layout: single
title: "CSS Flexbox와 Grid의 차이점"
categories: [til]
subcategory: CSS
---

> 2025-04-11 TIL - CSS Flexbox와 Grid의 차이점

# CSS Flexbox와 Grid의 차이점

**Flexbox**와 **Grid**는 페이지에서 **레이아웃을 구성할 때 자주 사용되는 CSS 속성**이다. 두 속성 모두 화면 요소를 배치하고 정렬하는 데 사용되지만, 각각의 특성과 목적에 따라 사용 방식에 차이가 있다.

## 1. 레이아웃 방향: 1차원 vs 2차원

- **Flexbox**는 **1차원 레이아웃**이다.
  - 주축(행 또는 열) 하나의 방향을 기준으로 요소를 정렬하고 배치하는 데 최적화되어 있따.
  - 단순한 수평 또는 수직 정렬에 효과적이다.
- **Grid**는 **2차원 레이아웃**이다.
  - 행과 열을 동시에 사용하여 요소를 배치할 수 있따.
  - 복잡한 레이아웃이나 페이지 전체 구조를 설계할 때 유용하다.

## 2. 사용 목적: 콘텐츠 중심 vs 레이아웃 중심

- **Flexbox**는 **콘텐츠 중심**의 레이아웃이다.
  - 콘텐츠가 추가되거나 줄어들 때 유연하게 대응한다.
  - 버튼 그룹, 내비게이션 바 등 한 줄의 콘텐츠가 주가 되는 구성에 적합하다.
- **Grid**는 **레이아웃 중심**의 레이아웃이다.
  - 명확하게 구분된 영역이나 구조를 설계할 때 적합하다.
  - 카드 레이아웃, 이미지 갤러리, 대시보드 구성 등에 효과적이다.

## 3. 기본 개념 및 동작 방식

- **Flexbox**는 요소들이 **자동 정렬**되는 구조이다.
  - `justify-content`, `align-items` 등의 속성을 사용해 주축 방향과 교차 축 방향의 정렬을 제어한다.
- **Grid**는 **격자(grid cell)** 기반으로 배치된다.
  - `grid-template-rows`, `grid-template-columns` 등을 활용해 행과 열의 크기를 정의하고 각 요소의 위치를 명확하게 지정한다.

## 요약 비교표

| 구분           | Flexbox                                           | Grid                                                           |
| -------------- | ------------------------------------------------- | -------------------------------------------------------------- |
| 레이아웃 방향  | 1차원 (row 또는 column)                           | 2차원 (row + column)                                           |
| 사용 목적      | 콘텐츠 중심 정렬                                  | 레이아웃 중심 구조                                             |
| 유용한 예시    | 버튼 그룹, 내비게이션 바                          | 카드 레이아웃, 갤러리, 대시보드                                |
| 요소 배치 방식 | 자동 정렬 (main/cross axis)                       | 셀 지정 배치 (grid cell)                                       |
| 주요 속성      | `display: flex`, `justify-content`, `align-items` | `display: grid`, `grid-template-columns`, `grid-template-rows` |

## 추가로 학습할 만한 자료

- [[1분코딩] CSS Flexbox와 Grid의 차이점 간략하게 정리해보기](https://www.youtube.com/watch?v=B1zuZDA4LOM)
- [[1분코딩] CSS Flexbox와 CSS Grid, 한번에 정리!](https://www.youtube.com/watch?v=eprXmC_j9A4)
