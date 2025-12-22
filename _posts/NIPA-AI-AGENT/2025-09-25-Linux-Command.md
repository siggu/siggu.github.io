---
layout: single
title: "리눅스 명령어 및 Shell Script"
categories: [nipa-ai-agent]
subcategory: Linux
---

> 2025-09-25 nipa-ai-agent-리눅스-명령어-및-Shell-Script

## 리다이렉션

이름 그대로 **방향을 바꾼다**는 의미이다. 특정 명령어의 입력이나 출력의 흐름을 기본 설정(키보드 입력, 화면 출력)이 아닌 다른 곳(파일)으로 변경하는 기능이다.

## 표준 스트림

- 표준 입력(stdin, 0): 기본적으로 키보드로부터의 입력을 받음
- 표준 출력(stdout, 1): 명령어의 정상적인 실행 결과를 화면에 출력
- 표준 에러(stderr, 2): 명령어 중 발생한 오류 메세지를 화면에 출력

## 주요 리다이렉션 기호

- `>` (출력 리다이렉션): 명령어의 표준 출력(stdout)을 파일로 보낸다. 파일이 존재하면 기존 내용을 덮어쓴다.
  ```bash
  # ls 명령어의 결과를 list.txt 파일에 저장합니다. (덮어쓰기)
  ls -l > list.txt
  ```
- `>>` (출력 리다이렉션 - 추가): 명령어의 표준 출력(stdout)을 파일의 기존 내용 끝에 추가한다.
  ```bash
  # 현재 날짜 정보를 date.txt 파일에 추가합니다.
  date >> date.txt
  ```
- `<` (입력 리다이렉션): 파일의 내용을 명령어의 표준 입력(stdin)으로 사용한다.
  ```bash
  # count.txt 파일의 단어 수를 셉니다. (키보드 입력 대신 파일 입력)
  wc -w < count.txt
  ```
- `2>` (에러 리다이렉션): 명령어의 표준 에러(stderr)를 파일로 보낸다.
  ```bash
  # find 명령어 실행 시 발생하는 오류를 error.log 파일에 저장합니다.
  find / -name "myfile" 2> error.log
  ```
- `&>` 또는 `2>&1` (출력/에러 리다이렉션): 표준 출력과 표준 에러를 모두 하나의 파일로 보낸다.
  ```bash
  # 명령어의 모든 결과(성공, 오류)를 result.log 파일에 저장합니다.
  command &> result.log
  # 위와 동일한 기능
  command > result.log 2>&1
  ```

# 파이프

파이프(`|`)는 두 개 이상의 명령어를 연결하는 **연결 통로**를 말한다. 첫 번째 명령어의 표준 출력(stdout)을 두 번째 명령어의 표준 입력(stdin)으로 직접 전달한다. 중간에 임시 파일을 만들 필요 없이 데이터의 흐름을 이어줄 수 있어 매우 유용하다.

## 파이프의 동작 방식

`command1 | command2`

1. `command1`이 실행되고, 그 결과가 화면에 출력되는 대신 파이프를 통해 `command2`로 전달된다.
2. `command2`는 `command1`의 결과를 자신의 입력으로 받아 작업을 수행한다.

## 파이프 사용 예시

```bash
# 현재 실행 중인 프로세스 목록에서 'bash'라는 단어가 포함된 라인만 필터링합니다.
ps -ef | grep bash

# 현재 디렉토리의 파일 목록을 긴 형식으로 출력하고, 그 결과를 less 명령어로 한 페이지씩 봅니다.
ls -l | less
```

# 필터

필터는 파이프와 함께 사용되어 데이터의 흐름 속에서 **필요한 데이터를 가공하거나 걸러내는 역할**을 하는 명령어들을 말한다. 표준 입력을 받아 처리한 후 표준 출력으로 결과를 내보내는 특징이 있어 파이프라인 중간에 위치하기에 적합하다.

## 주요 필터 명령어

- `grep`: 입력된 텍스트에서 **특정 패턴(문자열)을 포함하는 라인을 검색**하여 출력
  ```bash
  # access.log 파일에서 'ERROR'라는 단어가 포함된 모든 라인을 찾습니다.
  grep "ERROR" access.log
  ```
- `sort`: 입력된 텍스트 라인을 **정렬**한다. (알파벳, 숫자, 역순 등 다양한 옵션)
  ```bash
  # fruit.txt 파일의 내용을 알파벳 순으로 정렬합니다.
  sort fruit.txt
  ```
- `uniq`: 연속적으로 중복되는 라인을 제거한다.(보통 `sort`와 함께 사용)
  ```bash
  # fruit.txt를 정렬한 후 중복된 과일 이름을 제거합니다.
  sort fruit.txt | uniq
  ```
- `wc` (Word Count): 입력된 텍스트의 **라인, 단어, 문자 수**를 센다.
  ```bash
  # 현재 디렉토리에 몇 개의 파일/디렉토리가 있는지 확인합니다.
  ls | wc -l
  ```
- `head` / `tail`: 파일의 처음 또는 마지막 부분의 지정된 라인 수만큼 출력한다.
  ```bash
  # log.txt 파일의 마지막 10줄을 실시간으로 계속 확인합니다. (-f 옵션)
  tail -f log.txt
  ```
- `sed` (Stream Editor): 스트림(입력 데이터)을 편집하는 도구로, 주로 **문자열 치환**에 많이 사용
  ```bash
  # text.txt 파일의 'apple'을 'orange'로 바꾸어 화면에 출력합니다.
  sed 's/apple/orange/g' text.txt
  ```
- `awk`: 강력한 텍스트 처리 도구로, **열(column) 단위로 데이터를 추출하고 가공**하는 데 탁월하다.
  ```bash
  # 프로세스 목록에서 프로세스 ID(두 번째 열)만 출력합니다.
  ps -ef | awk '{print $2}'
  ```

# 리눅스서버 사용자 명령어

<aside>
💡

명령어 한 번에 여러 번 쓰려면 `;` 로 이어서 쓰기

```python
cd ..;ls -l;pwd
```

</aside>

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

## 파일 관리

```python
# 파일 및 디렉토리 목록
ls

# 옵션
-l (상세하게 출력), -la(숨김 파일까지 상세하게 출력),
-Fa (얜 뭐지), -r(reverse sort), -t(time), -s(size),
-LR(하위 폴더까지 출력), ...
```

### 디렉토리 관련

```python
# 현재 작업 디렉토리 확인
pwd

# 디렉토리 이동
cd $dir

# 디렉토리 생성
mkdir $dir
```

### 파일/디렉토리 삭제

```python
# 파일 삭제
rm $file

# 디렉토리+내용 삭제
rm -r $dir

# 강제 삭제(확인x)
rm -f $file

# 디렉토리 강제 삭제
rm -rf $dir
```

### 파일/디렉토리 복사/이동

```python
# 파일 복사
cp [원본_파일] [대상_위치]

# 심볼릭 링크 생성
ln -s $file $link

# 파일/디렉터리 이름 변경(dir1 -> dir2)
mv $dir1 $dir2

# 파일 디렉터리 이동
mv [원본_경로] [대상_경로]
```

### 파일 생성/내용 보기

```python
# 빈 파일 생성 또는 시간 업데이트
touch $file

# 파일 내용 전체 출력
cat $filename

# 파일에 내용 입력
cat > $filename

# 파일 내용 페이지별로 보기
more $file

# 파일 앞 10줄 보기
head #file

# 파일 뒤 10줄 보기
tail $file
```

### 용량/압축

```python
# 디렉토리 사용 용량 확인
du

# tar 아카이브 생성
tar cf $file.tar $dir

# tar 아카이브 압축 해제
tar xf $file.tar

# gzip 압축된 tar 생성
tar czf $file.tar

# gzip 압축된 tar 해제
tar xzf $file.tar

# tar 내용확인
tar tvf $file.tar

# gzip 압축
gzip $file

# 압축 해제
gzip -d
```

### 권한/소유권

```python
# 파일 권한 변경
chmod

# 하위 디렉토리까지 재귀적으로 적용
chmod -R

# 소유권 변경
chown $사용자:$그룹 $파일
```

### 검색/비교

```python
# 문자열 패턴 검색
grep

# 옵션
-r(recursive), -n(줄번호), -i(대소문자 무시)

# 특정 파일만 검색
grep --include="*.py"

# 두 파일 내용 비교
diff
```

### 프로그램 찾기

```python
# 실행 파일 경로 찾기
which $app

# 바이너리, 소스, 매뉴얼 위치 찾기
whereis $app
```

## 프로세스 관리

```python
# 현재 실행 중인 프로세스 출력
ps

# 옵션
-ef (모든 프로세스를 자세한 형태로 출력)
-elf (더 자세한 정보(긴 형태)

# 실시간 프로세스 모니터링(CPU, 메모리 사용량)
top

# 특정 프로세스 ID로 프로세스 종료
kill %1
```

### 작업 제어

```python
# 프로세스 확인
jobs

# 중단된 작업을 백그라운드로 실행
bg

# 백그라운드 작업을 포어그라운드로 가져오기
fg

# n번째 작업을 포어그라운드로 가져오기
fg n

# 명령어 뒤에 붙여 백그라운드 실행
python 3 ... &

# 로그아웃 후에도 프로세스가 계속 실행되도록 함
nohup python3 ...
```

### 키보드 단축기

```python
# 현재 실행 중인 프로세스 강제 종료
Ctrl + C

# 화면 출력 일시 정지
Ctrl + S

# 화면 출력 재개
Ctrl + Q

# 현재 프로세스 일시 중단(백그라운드로)
Ctrl + Z

# 입력 마치기
Ctrl + D
```

### 기타 유용한 명령어

```python
# 명령어 히스토리 보기
history

# 현재 날짜와 시간 출력
date

# 텍스트 출력(변수값 확인 등)
echo

# 환경 변수 설정/내보내기
export

# 명령어 별칭 만들기
alias

# 지정된 시간만큼 대기
sleep

# 주기적 작업 스케줄링
crontab

# 크론탭 옵션
-e (크론탭 만들기), -l (크론탭 리스트 보기)

```

Shell Script는 Shell이나 command line 인터프리터에서 구동 되도록 작성된 스크립트( windows의 batch와 동일)

확장자는 .sh 로 끝나며, 실행파일 모드여야한다.

```bash
#!/usr/bin/bash

printf "시작\n"
greeting="hello"
echo $greeting
echo $(which bash)

printf "단순 숫자 계산\n"
num1=10
num2=20
result=`expr $num1+$num2`
printf "%s\n" $result

printf "num1과 num2 비교\n"
if [ $num1 -gt $num2 ]; then
	echo num1 is bigger
else
	echo num2 is bigger
fi

printf "num1연산 결과와  num2  연산 결과 비교\n"
if (( $num1*num2 > $num2+100 ));then
	echo num1*num2 is larger
else
	echo num2+100 i larger
fi

printf "반복문\n"
# 반복문
for (( idx=1; idx<5; idx++));do
	echo $idx
	sleep 2
done

printf "반복문 2\n"
# 반복문-2
for idx2 in 5 10 15 20
do
	sleep $(( $idx2/5 ))
	echo $idx2
done
```

주의할 점은 변수를 선언할 때 조심해야 한다.

```bash
# 올바른 선언
num1=10

# 잘못된 선언
num1 = 10
```

`=` 사이를 띄우게 되면 변수 선언을 인식하지 못한다.

무언가를 계산해야 할 때 백틱(`) 과 `expr` 로 계산을 해준다.

```bash
result=`expr $num1+$num2`
```

실행 방법

1. `chmod 755 [실행할_파일]`
2. `bash [실행할_파일]`
