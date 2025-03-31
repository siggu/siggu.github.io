---
layout: single
title: "웹페이지 로딩 과정"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-03-30 TIL-웹페이지 로딩 과정

# 웹페이지 로딩 과정

예를 들어, 웹 브라우저에서 `"www.google.com"`을 입력하고 Enter를 누르면 아래와 같은 과정이 진행된다.

## 1. DNS 조회

- 사용자가 `"www.google.com"`을 입력하면, 브라우저는 먼저 이 도메인 이름을 IP 주소로 변환해야 한다. 이 과정을 **DNS 조회(DNS Lookup)** 라고 한다.

- 브라우저는 캐시된 DNS 기록을 먼저 확인하고, 없으면 로컬 DNS 서버에 요청하여 `"www.google.com"`에 해당하는 IP 주소를 얻는다.

## 2. TCP 연결 수립

- IP 주소가 확인되면, 브라우저는 서버와 TCP 연결을 수립한다.

> TCP(Transmission Control Protocol)는 데이터를 신뢰성 있게 전달하기 위한 프로토콜이다.

- 이 과정에서 브라우저는 서버와 3-way handshake를 수행한다.

## 3. HTTP 요청

- TCP 연결이 수립되면, 브라우저는 HTTP 또는 HTTPS 요청을 보낸다.

- 이 요청은 "GET /HTTP /1.1" 같은 형식으로, 웹 페이지를 요청하는 메시지이다. HTTPS를 사용할 경우, 이 단계에서 SSL/TSL handshake도 수행된다.

  - 이 과정에서는 브라우저와 서버가 암호화된 연결을 설정하기 위해 보안 인증서를 교환하고, 암호화 키를 협상한다.

## 4. 서버의 응답 받기

- 서버는 요청을 받고, 해당 리소스 (HTML, CSS, JavaScript, 이미지 등)를 브라우저에게 응답으로 보낸다.

  - 이 응답은 HTTP 응답 코드 (예: 200 OK)와 함께 전달된다.

## [5. 브라우저 렌더링 파이프라인](https://siggu.github.io/til/browser-rending-pipeline/)

- DOM과 CSSOM을 생성하고, 렌더 트리를 구성한 뒤, 레이아웃과 페인트 단계를 통해 웹 페이지가 화면에 표시된다.
