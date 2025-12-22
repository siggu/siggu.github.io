---
layout: single
title: "SQL 기초 및 PostgreSQL 연동"
categories: [nipa-ai-agent]
subcategory: SQL
---

> 2025-09-29 nipa-ai-agent-SQL-기초-및-PostgreSQL-연동

## HeidiSQL

HeidiSQL에서 아래와 같이 세션 생성(사용자, 암호는 Xshell에서 설정한대로 입력)

**public 우클릭 -> 트리 방식 옵션 -> 객체를 유형별로 묶기 체크**

## INSERT (데이터 삽입)

```sql
-- 기본 삽입
INSERT INTO student (NAME, AGE, dept, grade)
VALUES ('김정목', 26, '컴퓨터공학과', 'A');

-- 여러 행 삽입
INSERT INTO student (NAME, AGE, dept, grade) VALUES
('이영희', 21, '경영학', 'B'),
('박민수', 22, '컴퓨터공학', 'A'),
('최지원', 29, '수학', 'C'),
('정수진', 20, '경영학', 'B');
```

## UPDATE (데이터 수정)

```sql
-- 특정 학생의 학점 수정
UPDATE student SET grade = 'A'
WHERE NAME = '최지원';

-- 나이 증가
UPDATE student SET AGE = AGE + 1
WHERE dept = '컴퓨터공학';

-- 여러 칼럼 동시 수정
UPDATE student SET AGE = 21, grade = 'A'
WHERE NAME = '정수진';

-- 조건에 따른 update
UPDATE student SET grade='A', dept='%공학'
WHERE AGE=22;
```

## DELETE (데이터 삭제)

```sql
-- 조건에 따른 delete
DELETE FROM student WHERE AGE=100;
```

## SELECT (기본 조회)

```sql
-- 전체 조회
SELECT * FROM student;

-- 특정 컬럼만 조회
SELECT NAME, AGE, dept FROM student;

-- 조건부 조회
SELECT * FROM student WHERE AGE >= 20;
SELECT * FROM student WHERE dept = '컴퓨터공학';
SELECT * FROM student WHERE grade IN ('A', 'B');
```

```sql
-- 최소값/최대값만 조회
SELECT MIN(AGE) FROM student WHERE AGE > 20;
SELECT MAX(AGE) FROM student WHERE AGE < 30;

-- GROUP, COUNT
SELECT AGE, COUNT(*) FROM student GROUP BY age;

-- SUM
SELECT sum(AGE) FROM student WHERE AGE > 22;

-- AVG
SELECT avg(AGE) FROM student WHERE AGE > 25;
```

### WHERE 절 활용

```sql

-- 조건문 BETWEEN
SELECT * FROM student
WHERE AGE BETWEEN 22 AND 30;
SELECT * FROM student
WHERE NAME LIKE '김%'; -- LIKE 패턴 검색


SELECT * FROM student
WHERE dept LIKE '%공학%';
                -- AND, OR 조건
SELECT * FROM student
WHERE AGE >= 20 AND grade = 'A'

SELECT * FROM student
WHERE dept = '컴퓨터공학' OR
-- LKIE
SELECT * FROM student
WHERE dept LIKE '%공학';

-- 조건문 in (not in)
SELECT * FROM student
WHERE address IN ('서울', '경기');

-- BETWEEN 범위 검색
SELECT * FROM student
WHERE AGE BETWEEN 20 AND 25;
```

### 정렬과 제한

```sql
-- 정렬 (ORDER BY)
SELECT * FROM student
ORDER BY AGE ASC;

SELECT * FROM student
ORDER BY NAME DESC;

SELECT * FROM student
ORDER BY dept, AGE;

-- 개수 제한 (LIMIT)
SELECT * FROM student
ORDER BY AGE LIMIT 3;

SELECT * FROM student
ORDER BY regdate DESC LIMIT 5;
```

### JOIN

```sql
-- department 테이블 생성
CREATE TABLE department (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(50) UNIQUE NOT NULL,
    building VARCHAR(30),
    dean_name VARCHAR(30),
    phone VARCHAR(15)
);
```

```sql
-- 데이터 삽입
INSERT INTO department (dept_name, building, dean_name, phone) VALUES
('컴퓨터공학과', '공학관', '김컴퓨터', '02-1234-5001'),
('컴퓨터공학', '공학관', '김컴퓨터', '02-1234-5001'),
('생명공학과', '생명과학관', '이생명', '02-1234-5002'),
('생명공학', '생명과학관', '이생명', '02-1234-5002'),
('토목공학과', '공학관', '박토목', '02-1234-5003'),
('화학과', '자연과학관', '최화학', '02-1234-5004'),
('수학', '자연과학관', '정수학', '02-1234-5005'),
('경영학', '경영관', '윤경영', '02-1234-5006');
```

- student 테이블
  ```sql
  -- student 테이블
  idx |  name  | age |     dept     | grade |            regdate            | address
  -----+--------+-----+--------------+-------+-------------------------------+---------
     4 | 박철수 |  25 | 토목공학과   |       | 2025-09-29 14:44:35.020498+09 | 부산
     5 | 김훈이 |  27 | 생명공학과   |       | 2025-09-29 14:44:56.670409+09 | 천안
     2 | 홍길동 |  22 | 생명공학     | A     | 2025-09-29 14:44:09.835674+09 | 서울
     3 | 김영희 |  22 | 화학과       | A     | 2025-09-29 14:44:19.989548+09 | 경기
     8 | 김정목 |  26 | 컴퓨터공학과 | A     | 2025-09-29 15:07:18.969215+09 | 서울
    10 | 박민수 |  23 | 컴퓨터공학   | A     | 2025-09-29 15:07:18.983287+09 | 인천
     7 | 김정목 |  26 | 컴퓨터공학과 | A     | 2025-09-29 15:05:55.523554+09 | 광주
    12 | 정수진 |  21 | 경영학       | A     | 2025-09-29 15:07:18.983287+09 | 대구
     9 | 이영희 |  21 | 경영학       | B     | 2025-09-29 15:07:18.983287+09 | 대전
    11 | 최지원 |  29 | 수학         | A     | 2025-09-29 15:07:18.983287+09 | 제주도
  ```
- department 테이블
  ```sql
  -- department 테이블
   dept_id |  dept_name   |  building  | dean_name |    phone
  ---------+--------------+------------+-----------+--------------
         1 | 컴퓨터공학과 | 공학관     | 김컴퓨터  | 02-1234-5001
         2 | 컴퓨터공학   | 공학관     | 김컴퓨터  | 02-1234-5001
         3 | 생명공학과   | 생명과학관 | 이생명    | 02-1234-5002
         4 | 생명공학     | 생명과학관 | 이생명    | 02-1234-5002
         5 | 토목공학과   | 공학관     | 박토목    | 02-1234-5003
         6 | 화학과       | 자연과학관 | 최화학    | 02-1234-5004
         7 | 수학         | 자연과학관 | 정수학    | 02-1234-5005
         8 | 경영학       | 경영관     | 윤경영    | 02-1234-5006
  ```

1. **INNER JOIN**

   ```sql
   -- 학생과 학과 정보 (매칭되는 데이터만)
   SELECT s.name, s.age, s.dept, d.building, d.phone
   FROM student s
   INNER JOIN department d ON s.dept = d.dept_name;

     name  | age |     dept     |  building  |    phone
   --------+-----+--------------+------------+--------------
    박철수 |  25 | 토목공학과   | 공학관     | 02-1234-5003
    김훈이 |  27 | 생명공학과   | 생명과학관 | 02-1234-5002
    홍길동 |  22 | 생명공학     | 생명과학관 | 02-1234-5002
    김영희 |  22 | 화학과       | 자연과학관 | 02-1234-5004
    김정목 |  26 | 컴퓨터공학과 | 공학관     | 02-1234-5001
    박민수 |  23 | 컴퓨터공학   | 공학관     | 02-1234-5001
    김정목 |  26 | 컴퓨터공학과 | 공학관     | 02-1234-5001
    정수진 |  21 | 경영학       | 경영관     | 02-1234-5006
    이영희 |  21 | 경영학       | 경영관     | 02-1234-5006
    최지원 |  29 | 수학         | 자연과학관 | 02-1234-5005
   ```

2. **LEFT JOIN**

   ```sql
   -- 모든 학생 정보 + 학과 정보 (학과 정보가 없는 학생도 포함)
   SELECT s.name, s.age, s.dept, d.building
   FROM student s
   LEFT JOIN department d ON s.dept = d.dept_name
   ORDER BY s.name;

     name  | age |     dept     |  building
   --------+-----+--------------+------------
    김영희 |  22 | 화학과       | 자연과학관
    김정목 |  26 | 컴퓨터공학과 | 공학관
    김정목 |  26 | 컴퓨터공학과 | 공학관
    김훈이 |  27 | 생명공학과   | 생명과학관
    박민수 |  23 | 컴퓨터공학   | 공학관
    박철수 |  25 | 토목공학과   | 공학관
    이영희 |  21 | 경영학       | 경영관
    정수진 |  21 | 경영학       | 경영관
    최지원 |  29 | 수학         | 자연과학관
    홍길동 |  22 | 생명공학     | 생명과학관
   ```

3. **RIGHT JOIN**

   ```sql
   -- 모든 학과 정보 + 학생 정보 (학생이 없는 학과도 포함)
   SELECT d.dept_name, d.building, s.name, s.age
   FROM student s
   RIGHT JOIN department d ON s.dept = d.dept_name
   ORDER BY d.dept_name;

     dept_name   |  building  |  name  | age
   --------------+------------+--------+-----
    수학         | 자연과학관 | 최지원 |  29
    경영학       | 경영관     | 정수진 |  21
    경영학       | 경영관     | 이영희 |  21
    화학과       | 자연과학관 | 김영희 |  22
    생명공학     | 생명과학관 | 홍길동 |  22
    생명공학과   | 생명과학관 | 김훈이 |  27
    컴퓨터공학   | 공학관     | 박민수 |  23
    토목공학과   | 공학관     | 박철수 |  25
    컴퓨터공학과 | 공학관     | 김정목 |  26
    컴퓨터공학과 | 공학관     | 김정목 |  26
   ```

4. **FULL OUTER JOIN**

   ```sql
   -- 학생과 학과 모든 정보 (매칭되지 않는 것도 모두 포함)
   SELECT s.name, s.dept as student_dept, d.dept_name, d.building
   FROM student s
   FULL OUTER JOIN department d ON s.dept = d.dept_name
   ORDER BY s.name;

     name  | student_dept |  dept_name   |  building
   --------+--------------+--------------+------------
    김영희 | 화학과       | 화학과       | 자연과학관
    김정목 | 컴퓨터공학과 | 컴퓨터공학과 | 공학관
    김정목 | 컴퓨터공학과 | 컴퓨터공학과 | 공학관
    김훈이 | 생명공학과   | 생명공학과   | 생명과학관
    박민수 | 컴퓨터공학   | 컴퓨터공학   | 공학관
    박철수 | 토목공학과   | 토목공학과   | 공학관
    이영희 | 경영학       | 경영학       | 경영관
    정수진 | 경영학       | 경영학       | 경영관
    최지원 | 수학         | 수학         | 자연과학관
    홍길동 | 생명공학     | 생명공학     | 생명과학관
   ```

5. **집계와 함께 사용**

   ```sql
   -- 학과별 학생 수
   SELECT d.dept_name, d.building, COUNT(s.idx) AS student_count
   FROM department d
   LEFT JOIN student s ON d.dept_name = s.dept
   GROUP BY d.dept_id, d.dept_name, d.building
   ORDER BY student_count DESC;

     dept_name   |  building  | student_count
   --------------+------------+---------------
    컴퓨터공학과 | 공학관     |             2
    경영학       | 경영관     |             2
    생명공학     | 생명과학관 |             1
    화학과       | 자연과학관 |             1
    생명공학과   | 생명과학관 |             1
    수학         | 자연과학관 |             1
    컴퓨터공학   | 공학관     |             1
    토목공학과   | 공학관     |             1
   ```

## Vector 조회

```sql
-- 개인 db로 접속
root@edu007-20250924-1450:/# psql -U test -d testdb
```

```sql
-- 테이블 생성
create table prod_embeddings(id serial primary key, name text, embedding VECTOR(3));
```

```sql
-- vector 데이터 삽입
insert into prod_embeddings (name, embedding) values ('Product A', '[0.1, 0.2, 0.3]'), ('Product B', '[0.5, 0.6, 0.1]'), ('Product C', '[0.9, 0.8, 0.7]');
INSERT 0 3
testdb=# select * from prod_embeddings;
 id |   name    |   embedding
----+-----------+---------------
  1 | Product A | [0.1,0.2,0.3]
  2 | Product B | [0.5,0.6,0.1]
  3 | Product C | [0.9,0.8,0.7]
```

```sql
-- vector 데이터 조회
-- 주어진 vector와 가장 유사한 vector를 반환(limit 1)
select id, name, embedding <=> '[0.2, 0.3, 0.4]' as similarity
from prod_embeddings order by similarity limit 1;
 id |   name    |      similarity
----+-----------+----------------------
  1 | Product A | 0.007416616119754305

-- 주어진 vector와 유사한 vector들을 유사도를 기준으로 정렬
select id, name, embedding <=> '[0.2, 0.3, 0.4]' as similarity
from prod_embeddings order by similarity;
 id |   name    |      similarity
----+-----------+----------------------
  1 | Product A | 0.007416616119754305
  3 | Product C |  0.06674965407913813
  2 | Product B |  0.24533343867126223

-- 유사도 기준 정렬 및 embedding 정보 추가
select id, name, embedding, embedding <=> '[0.2, 0.3, 0.4]' as similarity
from prod_embeddings order by similarity;
 id |   name    |   embedding   |      similarity
----+-----------+---------------+----------------------
  1 | Product A | [0.1,0.2,0.3] | 0.007416616119754305
  3 | Product C | [0.9,0.8,0.7] |  0.06674965407913813
  2 | Product B | [0.5,0.6,0.1] |  0.24533343867126223
```

- 벡터 코사인 유사도 공식:
  두 벡터의 내적을 각 벡터의 크기(L2 Norm)의 곱으로 나눈 값 cos(θ) = (A · B) / (||A|| \* ||B||)

> **Xshell에서 postgres 이외의 db에 접속하는 법**

```sql
psql -U test -d testdb
```

컨테이너 내부에서 위와 같이 적으면 된다.

- test: 접속하려는 db명
- testdb: postgres의 db명

## Python에서 PostgreSQL 연동

Xshell에서 테스트를 위한 디렉토리 생성 후 `pip install psycopg2-binary` 입력

```bash
(myvenv) edu007@vultr:~/work$ mkdir psycopg_test
(myvenv) edu007@vultr:~/work$ cd psycopg_test/
(myvenv) edu007@vultr:~/work/psycopg_test$ pip install psycopg2-binary
Collecting psycopg2-binary
  Downloading psycopg2_binary-2.9.10-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 12.3 MB/s eta 0:00:00
Installing collected packages: psycopg2-binary
Successfully installed psycopg2-binary-2.9.10
```

```python
import psycopg2
from datetime import datetime

# postgres connection infomation
db_config = {
  'host': '161.118.158.91',
  'port': 5432,
  'database': 'testdb',
  'user': 'test',
  'password': '1234'
}

try:
  # DB 연결
  conn = psycopg2.connect(**db_config)
  cursor = conn.cursor()

  # 테이블이 존재하지 않는 경우 새로 생성
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS mytest(
      id serial primary key,
      title varchar,
      doc_body varchar,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  """)
  conn.commit()

  # insert database
  sample_data = [
    ("첫 번째 제목", "첫 번째 문서의 본문 내용"),
    ("두 번째 제목", "python으로 PostgreSQL을 작성하는 테스트"),
    ("세 번째 제목", "데이터베이스 프로그래밍 테스트")
  ]

  # 데이터 삽입 쿼리문
  insert_query = """
    INSERT INTO mytest (title, doc_body)
    VALUES (%s, %s)
    RETURNING id;
  """

  inserted_ids = []
  for title, doc_body in sample_data:
    cursor.execute(insert_query, (title, doc_body))
    inserted_id = cursor.fetchone()[0]
    inserted_ids.append(inserted_id)
    print("inserted data id: ", inserted_id)
  conn.commit()

  # mytest 테이블에서 id/title/doc_body/created_at 들을 열로 하여 최대 10개 데이터를 가져와 id를 기준으로 내림차순 해라.
  cursor.execute("""
    select id, title, doc_body, created_at from mytest order by id desc limit 10
  """)
  rows = cursor.fetchall()
  print("최근 10개 데이터: ")
  for row in rows:
    print(row)

except Exception as e:
  print("Error: ", e)

finally:
  if cursor:
    cursor.close()
  if conn:
    conn.close();
```

### 실행 결과

```bash
(myvenv) edu007@vultr:~/work/psycopg_test$ python postgresql_practice.py
inserted data id:  4
inserted data id:  5
inserted data id:  6
최근 10개 데이터:
(6, '세 번째 제목', '데이터베이스 프로그래밍 테스트', datetime.datetime(2025, 9, 29, 17, 23, 18, 850777))
(5, '두 번째 제목', 'python으로 PostgreSQL을 작성하는 테스트', datetime.datetime(2025, 9, 29, 17, 23, 18, 850777))
(4, '첫 번째 제목', '첫 번째 문서의 본문 내용', datetime.datetime(2025, 9, 29, 17, 23, 18, 850777))
(3, '세 번째 제목', '데이터베이스 프로그래밍 테스트', datetime.datetime(2025, 9, 29, 17, 22, 7, 762302))
(2, '두 번째 제목', 'python으로 PostgreSQL을 작성하는 테스트', datetime.datetime(2025, 9, 29, 17, 22, 7, 762302))
(1, '첫 번째 제목', '첫 번째 문서의 본문 내용', datetime.datetime(2025, 9, 29, 17, 22, 7, 762302))
```

3개씩 두 번 들어간 이유는 맨 처음 실행했을 때 insert query 이후에 오류가 있어서 다시 실행함
