---
layout: single
title: "리눅스 서버 관리자 명령"
categories: [nipa-ai-agent]
subcategory: Linux
---

## 운영체제 관리

```bash
# 현재 어디 서버에 있는지 확인
uname -a
```

```bash
lsb_release -a
```

```bash
# systemctl
sudo systemctl start nginx
sudo systemctl status nginx
sudo systemctl stop nginx
sudo systemctl status nginx
```

```bash
service
```

## 사용자 관리 및 로그인/로그아웃

```bash
sudo adduser $user_id
```

```bash
sudo deluser $user_id
```

```bash
logout / exit
```

공용 서버에서 개인 서버로 이동하기 위해서는 ssh키가 필요하다.

때문에 Xftp 이용해서 ssh 키를 공용 서버 폴더(ex: .ssh)에 옮겨주자.

```bash
# ssh: 원격 명령
# 개인 -> 공용 이동(ssh 키 필요 없는 경우)
ssh ~~-p 10022~~ edu007@158.247.210.130

# 공용 -> 개인(ssh 키 필요한 경우)
# 축약 버전
ssh -i ~/.ssh/ssh-key*007.key ubuntu@161.118.158.91
# 풀버전
ssh -i ~/.ssh/ssh-key-2025-09-24-007.key ubuntu@161.118.158.91
```

```bash
# 사용자 권한 부여
usermod -a -G sudo #user_id

# 비밀번호 변경
passwd $user_id

# 같은 세션에 누가 있는지 확인
who
```

### scp

```bash
# 원격 서버로 파일 주기(업로드)
scp [옵션] [보낼_파일] [원격사용자]@[원격서버주소]:[받을_경로]

# 원격 서버에서 파일 가져오기(다운로드)
scp [옵션] [원격사용자]@[원격서버주소]:[가져올_파일] [저장할_경로]
```

```bash
# 파일 전송하기(개인->공용)
scp /home/ubuntu/test/download_me.txt edu007@158.247.210.130:~/test/

# 파일 전송하기(공용->개인)
scp -i ~/.ssh/ssh-key-2025-09-24-007.key /home/edu007/test/download_me.txt ubuntu@161.118.158.91:~/test/
```

```bash
# 파일 받아오기(개인에서 공용파일)
scp edu007@158.247.210.130:~/test/download_me.txt /home/ubuntu/test/

# 파일 받아오기(공용에서 개인파일)
scp -i ~/.ssh/ssh-key-2025-09-24-007.key ubuntu@161.118.158.91:~/test/download_me.txt /home/edu007/test/
```

## 패키지 설치

```bash
apt

# ubuntu의 경우
apt-get

# redhat 계열인 경우
redhat
```

```bash
apt update
apt install $패키지명
apt remove $패키지명
apt purge $패키지명
apt -only-upgrade install $패키지명
```

```bash
apt list
apt search
apt show
```

```bash
rpm -ivh $설치파일.rpm
```

## 네트워크 관리

```bash
ping $ip주소
telnet $ip주소 $접속포트
ifconfig
ip a
```

```bash
netstat
netstat -nlpt
netstat -lptu
```

```bash
curl
```

```bash
wget
```

### 시간대 변경

```bash
sudo timedatectl set-timezone Asia/Seoul
```
