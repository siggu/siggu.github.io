---
layout: single
title: "Docker 명령어 및 사용법"
categories: [nipa-ai-agent]
subcategory: Docker
---

## docker란?

- 리눅스 서버 내에서 구동되는 가상 서버
- 구동되는 SW마다 구동을 위해 필요한 라이브러리,환경,커널버전 등이 모두 다름. → 동일 호스트에서 여러가지 SW를 설치하고 구동할때, library가 충돌하거나, OS와 호환이 되지 않아 설치 및 구동이 안되는 현상이 발생.
  이를 해결하기 위해 특정 SW와 해당 SW가 구동하기위해 필요한 가상의 OS환경을 하나의 package안에 통합하고, 이를 마치 독립된 호스트처럼 실행함, 이에따라 동일한 호스트에서 서로 다른 환경에서 구동하는 다양한 SW들을 구동할 수 있음.
- VM과의 차이: VM의 경우 개별 VM이 Host OS내에서 자신의 가상화된 HW+OS+자원을 모두 보유( 자유도가 높은 대신 많은 자원이
  필요)
- docker: host OS의 운영체제 까지 공유함. docker engine 위에서 각각의 가상화 서버(dockr컨테이너)들은 SW구동 환경만 독립적으로 보유 → 높은 자원효율
- docker 컨테이너: 실제 가상 host가 되어 SW를 구동하고 사용자의 데이터를 보관하는 가상환경
- docker image: 컨테이너를 생성하기 위한 원본 이미지 파일

# 유용한 docker 명령어

### 이미지 관리

```bash
# 현재 설치된 이미지 목록 확인
docker images

# 특정 이미지 다운로드
docker image pull $이미지명

# 특정 이미지 삭제
docker rmi $이미지명
```

### 컨테이너 관리

```bash
# 현재 구동중인 컨테이너 정보 확인
docker container ls

# 모든 컨테이너 확인 (중지된 것 포함)
docker container ls -a

# 컨테이너 시작
docker container start $컨테이너ID

# 컨테이너 중지
docker container stop $컨테이너ID

# 컨테이너 삭제
docker container rm $컨테이너ID
```

## 컨테이너 조작 명령어

### 컨테이너 접속 및 파일 복사

```bash
# 구동중인 컨테이너 내부에 접속
docker exec -it $컨테이너ID /bin/bash

# 호스트에서 컨테이너로 파일 복사
docker cp $파일 $컨테이너ID:$컨테이너_디렉토리
# 예시: docker cp /home/test001/mytest.txt 64c55b41e2fd:/tmp

# 컨테이너에서 호스트로 파일 복사
docker cp $컨테이너ID:$컨테이너_디렉토리/파일 호스트디렉토리_파일
# 예시: docker cp 64c55b41e2fd:/tmp/mytest001.txt /tmp/mytest002.txt
```

- 호스트에서 컨테이너로 파일 복사 예시
  ```bash
  root@edu007-20250924-1450:~/temp# nano docker_test.txt
  root@edu007-20250924-1450:~/temp# cat docker_test.txt
  ddddddddddddddddd
  aaaaaaaaaaaaaaa
  bbbbbbbbbbbbbbbbbb
  dxxxxxxxxxxxxx
  root@edu007-20250924-1450:~/temp# docker container ls
  CONTAINER ID   IMAGE                    COMMAND                  CREATED        STATUS          PORTS     NAMES
  1d0965cef543   pgvector/pgvector:pg17   "docker-entrypoint.s…"   26 hours ago   Up 12 minutes             pg-postgres
  root@edu007-20250924-1450:~/temp# docker container ls
  CONTAINER ID   IMAGE                    COMMAND                  CREATED        STATUS          PORTS     NAMES
  1d0965cef543   pgvector/pgvector:pg17   "docker-entrypoint.s…"   26 hours ago   Up 12 minutes             pg-postgres
  root@edu007-20250924-1450:~/temp# docker cp docker_test.txt 1d0965cef543:/tmp
  Successfully copied 2.05kB to 1d0965cef543:/tmp
  root@edu007-20250924-1450:~/temp# docker exec -it 1d0965cef543 /bin/bash
  root@edu007-20250924-1450:/# ls
  bin  boot  dev	docker-entrypoint-initdb.d  etc  home  lib  lib64  media  mnt  opt  proc  root	run  sbin  srv	sys  tmp  usr  var
  root@edu007-20250924-1450:/# cd tmp/
  root@edu007-20250924-1450:/tmp# ls
  docker_test.txt
  root@edu007-20250924-1450:/tmp# cat docker_test.txt
  ddddddddddddddddd
  aaaaaaaaaaaaaaa
  bbbbbbbbbbbbbbbbbb
  dxxxxxxxxxxxxx
  ```
- 컨테이너에서 호스트로 파일 복사
  ```bash
  root@edu007-20250924-1450:/tmp# cat > docker_test2.txt
  this is database backup
  데이터베이스 백업파일
  root@edu007-20250924-1450:/tmp# cat docker_test2.txt
  this is database backup
  데이터베이스 백업파일
  root@edu007-20250924-1450:/tmp# exit
  exit
  root@edu007-20250924-1450:~/temp# ls -l
  total 28
  -rw-r--r-- 1 root root   68 Sep 25 17:37 docker_test.txt
  drwxr-xr-x 2 root root 4096 Sep 25 15:40 test
  -rw-r--r-- 1 root root  146 Sep 25 15:04 test001.py
  -rw-r--r-- 1 root root   87 Sep 25 15:18 test002.py
  -rw-r--r-- 1 root root   17 Sep 25 15:19 test002_result.txt
  -rw-r--r-- 1 root root   17 Sep 25 15:20 test002_result_2.txt
  -rw-r--r-- 1 root root   17 Sep 25 15:21 test002_result_3.log
  -rwxrwxrwx 1 root root    0 Sep 25 14:31 test1.txt
  root@edu007-20250924-1450:~/temp# docker cp 1d0965cef543:/tmp/docker_test2.txt /temp
  Successfully copied 2.05kB to /temp
  root@edu007-20250924-1450:~/temp# ls -l
  total 28
  -rw-r--r-- 1 root root   68 Sep 25 17:37 docker_test.txt
  drwxr-xr-x 2 root root 4096 Sep 25 15:40 test
  -rw-r--r-- 1 root root  146 Sep 25 15:04 test001.py
  -rw-r--r-- 1 root root   87 Sep 25 15:18 test002.py
  -rw-r--r-- 1 root root   17 Sep 25 15:19 test002_result.txt
  -rw-r--r-- 1 root root   17 Sep 25 15:20 test002_result_2.txt
  -rw-r--r-- 1 root root   17 Sep 25 15:21 test002_result_3.log
  -rwxrwxrwx 1 root root    0 Sep 25 14:31 test1.txt
  ```

### 기타 명령어

```bash
# 볼륨 관리
docker volume

# 컨테이너 실행(docker container 생성하고 시작 -> docker-compose)
docker run

# PostgreSQL에 연결
psql -U postgres
```

## Docker Compose 명령어

```bash
# Docker 이미지를 Docker 호스트로 빌드 및 구동
docker-compose

# 빌드만 실행
docker-compose build

# 백그라운드에서 실행
docker-compose up -d

# 컨테이너 정리 및 제거 (이미지, 볼륨, 고아 컨테이너 모두 제거)
docker-compose down --rmi all --volumes --remove-orphans
```
