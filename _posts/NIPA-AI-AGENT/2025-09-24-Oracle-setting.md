---
layout: single
title: "Oracle 실습 환경 구성"
categories: [nipa-ai-agent]
subcategory: Web
---

## 클라우드 와 vm 와 docker 와 Linux

### 가상머신(Virtual Machine)

- 하나의 물리적인 컴퓨터에 여러개의 가상 머신을 생성
- Host OS: 실제 물리적인 서버
- Guest OS: 실제 물리적인 서버에 기생하는 가상 서버
- 하이퍼바이저: 물리적인 하나의 머신위에 여러개의 가상머신을 구동/관리( vmware,kvm)
- 장점: guestOS가 하나의 완벽한 독립서버처럼 구동(실제와 구분불가, 독립된 컴퓨터처럼 사용가능)
- 단점: guestOS들이 Host OS의 자원을 배타적으로 사용(자원낭비/자원부족현상)

→ 클라우드: 가상머신을 임대하여 장사

### 도커 (docker)

- 하나의 물리적인 컴퓨터에 여러개의 가상 머신을 생성
- Host OS: 실제 물리적인 서버
- 컨테이너(Guest-OS): 실제 물리적인 서버에 기생하는 가상 서버, 단 host-OS를 공유
- 도커(=하이퍼바이저): 물리적인 하나의 머신위에 여러개의 가상환경을 구동
- 장점: 컨테이너들이 Host-OS의 자원을 Share(부족하면 끌어다쓰거나, 반납할수 있다)
- 단점: 컨테이너(게스트OS)가 완전한 Server 가 아님(컴퓨터처럼 자유롭게 활용 못함)

## OCI 환경 구성

### 필요한 sw tool 설치 후 서버에 접속

#### xshell, xftp 설치

https://www.netsarang.com/ko/free-for-home-school/

### 가상환경 연결

```bash
source myvenv/bin/activate
```

```bash
conda activate venv
```

### Xshell로 생성한 서버에 ssh 접속하기

호스트: 할당받은 IP주소
사용자이름: Ubuntu
public key 선택 → 설정 → 키파일 지정 → [가져오기 ] → OCI 인스턴스 생성시 download한 SSH key파일을 지정

### 서버에 ssh로 접속 후 필요한 패키지를 설치[개인용]

```
[1] Linux update
sudo apt-get update
sudo apt-get -y upgrade
sudo apt install net-tools

[2] nginx 설치
sudo apt install nginx

[3] 서버 방화벽 오픈
```

# 우분투 표준

ufw allow 8080/tcp

# OCI 표준

`sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
80,5000,3389,5432` 등 필요한 tcp port 오픈

[4] docker 설치

```
apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose
sudo systemctl status docker
```

### 서버에 ssh로 접속 후 필요한 설정을 추가[개인용 + 공유 작업용]

[5] xrdp설치

```
sudo apt install xfce4 xfce4-goodies xorg dbus-x11 x11-xserver-utils
sudo apt install xrdp
sudo systemctl enable xrdp
sudo adduser xrdp ssl-cert
sudo ufw allow 3389/tcp
sudo systemctl enable xrdp
sudo apt-get install python3-tk
```

[6] python 기상환경 설치( venv 기반)

```
sudo apt install python3-venv
sudo apt install python3-pip
```

[개인 + 공유 작업용] `python3 -m venv myvenv`

[7] python 가상환경 설치 ( Anacodna 기반)

```
sudo apt install curl -y
curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash anaconda.sh
```

### 아나콘다 설치

https://www.anaconda.com/download

설치 후 propmt 입장

```bash
conda create -n "가상환경이름" python=버전
```

```bash
conda create -n venv python=3.10
```

> cmd에서 찾지 말고 anaconda propmt에서 찾자.

### 서버에 ssh로 접속 후 database를 설치

### 루트 폴더에 `docker-compose.yml`파일을 생성

```bash
# 루트 디렉토리에 직접 파일 생성
sudo nano /root/docker-compose.yml
```

#### 1단계: 파일 내용 확인

```bash
cat docker-compose.yml
```

#### 2단계: 파일이 비어있다면 작성

```bash
nano docker-compose.yml
```

```bash
version: '3.4'
services:
  db_singlepoint:
    image: pgvector/pgvector:pg17
    container_name: "pg-postgres"
    environment:
      - "POSTGRES_PASSWORD=dbtest1212"
      - "TZ=Asia/Seoul"
    volumes:
      - "postgres_data:/var/lib/postgresql/data"
      - "./init:/docker-entrypoint-initdb.d"
    restart: unless-stopped
    network_mode: "host"
volumes:
  postgres_data:
```

#### 저장 후 실행

```bash
*# Ctrl + O → Enter → Ctrl + X (nano 저장 및 종료)
docker-compose up -d*
```

### 설치된 docker 기반 postgresql에 접속

```
docker container ls

docker exec –it 컨테이너ID /bin/bash

(( 컨테이너 내부 shell 환경에서 ))
psql -U postgres
create user test01 password ‘test1234’ superuser;
create database testdb;
grant all privileges on database testdb to test01;
CREATE EXTENSION IF NOT EXISTS vector;
```

### 생성된 database에 heidiSQL로 접속

- 네트워크 유형: PostgresSQL(TCP/IP)

  - 라이브러리: libq-15.dll
  - 호스트명: 자신이 생성한 서버 IP
  - 사용자: postgres
  - 암호: dbtest1212
  - 포트: 5432
