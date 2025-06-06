---
layout: single
title: "이미지 크기로 인한 렌더링 속도 저하 개선 방법"
categories: [til]
subcategory: Web
---

> 2025-04-13 TIL-이미지 크기로 인한 렌더링 속도 저하 개선 방법

# 이미지 크기로 인한 렌더링 속도 저하 개선 방법

이미지 크기가 클 경우 렌더링 속도가 느려질 수 있다. 이를 개선하기 위한 방법은 크게 **네 가지**로 나눌 수 있다.

### 1. **이미지 포맷 최적화**

전통적인 JPEG나 PNG 대신, 압축 효율이 뛰어난 **WebP**나 **AVIF**와 같은 최신 이미지 포맷을 사용하는 방법이다. 이 포맷들은 이미지 품질을 거의 유지하면서도 파일 크기를 대폭 줄여준다.

- **AVIF**: 가장 높은 압축률을 자랑하지만, 지원하는 브라우저가 상대적으로 적음

- **WebP**: 대부분의 최신 브라우저에서 지원하며, JPEG보다 약 30% 더 작은 파일 크기

📌 **주의**: 일부 구버전 브라우저에서는 WebP나 AVIF를 지원하지 않으므로, **호환성**을 고려한 대체 이미지 설정이 필요하다.

### ✅ 호환성 해결 방법: `<picture>` 요소 사용

```html
<picture>
  <source srcset="image.avif" type="image/avif" />
  <source srcset="image.webp" type="image/webp" />
  <img src="image.jpg" alt="Image description" />
</picture>
```

- 브라우저는 지원 가능한 형식을 자동으로 선택해 로드한다.

- 가장 우선순위가 높은 `AVIF` → `WebP` → 기본 `JPG` 순으로 시도된다.

## 2. **이미지 크기 조정 및 Responsive Images**

렌더링되는 크기보다 이미지 자체가 과도하게 클 경우, 불필요한 리소스 낭비가 발생한다. 이를 방지하기 위해 **이미지를 적절한 크기로 리사이징**하고, **디바이스 해상도에 따라 다른 이미지를 제공**하는 방식이 필요하다.

- 서버 또는 빌드 단계에서 리사이징 처리

- `<img>` 태그에 `srcset`과 `sizes` 속성 활용

  ```html
  <img
    src="image-800.jpg"
    srcset="image-400.jpg 400w, image-800.jpg 800w, image-1200.jpg 1200w"
    sizes="(max-width: 600px) 400px, (max-width: 1024px) 800px, 1200px"
    alt="Responsive example"
  />
  ```

- **srcset**: 다양한 해상도의 이미지 제공

- **sizes**: 현재 뷰포트에 따라 어느 크기의 이미지를 사용할지 브라우저에 힌트를 제공

## 3. **지연 로딩 (Lazy Loading)**

초기 로딩 시 모든 이미지를 한 번에 로드하는 대신, **사용자가 해당 위치로 스크롤할 때 이미지가 로드**되도록 설정하는 방식이다.

- 기본적인 구현 방법:

  ```html
  <img src="image.jpg" alt="..." loading="lazy" />
  ```

- 이를 통해 초기 페이지 로딩 속도를 줄이고, 불필요한 리소스 낭비를 방지할 수 있다.

📌 **주의**:

- **첫 화면(above the fold)**에 나오는 이미지는 lazy loading 대상에서 제외하는 것이 좋다.

- SEO 최적화가 중요한 콘텐츠에서는 이미지 인덱싱이 안 될 수 있으므로 주의해야 한다.

### 4. **CDN(Content Delivery Network) 활용**

이미지를 **CDN에 업로드하여 전 세계 여러 위치에 분산 배포**하면, 사용자는 지리적으로 가까운 서버에서 이미지를 받아오게 된다. 이를 통해 로딩 속도가 빨라지고 서버 부하도 줄일 수 있다.

추가적으로:

- 자주 요청되는 이미지는 **CDN 캐시**에 저장되어, 반복 요청 시 훨씬 더 빠르게 응답할 수 있다.

- 이미지 변형 기능(리사이징, 포맷 변환 등)을 제공하는 CDN을 이용하면 백엔드 없이도 다양한 사이즈 대응이 가능하다.

## ✅ 마무리 정리

이미지 렌더링 최적화를 위해 다음 네 가지 전략을 조합하여 사용하는 것이 효과적이다:

1. **WebP, AVIF 등의 포맷으로 변환**

2. **디바이스에 맞는 사이즈로 리사이징 및 Responsive Images 적용**

3. **지연 로딩을 통해 초기 로딩 속도 개선**

4. **CDN을 통해 전송 속도 및 서버 부하 개선**

이러한 방법들을 상황에 맞게 조합하면, 이미지 로딩 속도를 효과적으로 개선하고 전반적인 사용자 경험을 크게 향상시킬 수 있다.
