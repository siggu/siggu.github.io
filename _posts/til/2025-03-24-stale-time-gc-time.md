---
layout: single
title: "staleTime과 gcTime"
categories: [til]
# tags: ["머신러닝", "웹앱"]
---

> 2025-03-24 TIL-staleTime과 gcTime

# staleTime과 gcTime

## `staleTime` (데이터 신선도 시간)

- 설명: `staleTime`은 데이터가 "신선한(fresh)" 상태로 유지되는 시간을 말한다.

- 기본값: `0ms` (즉시 stale)

- 동작 방식:

  - `staleTime` 동안은 캐시된 데이터가 최신 상태로 간주된다.

  - `staleTime`이 지나면 데이터는 "stale(오래된)" 상태가 되어 다시 `fetch`할 수 있다.

  - 하지만 **자동으로 다시 fetch되지는 않고,** `refetch`가 발생해야 데이터를 다시 가져온다.

### 예제

```tsx
useQuery({
  queryKey: ["posts"],
  queryFn: fetchPosts,
  staleTime: 1000 * 60 * 5, // 5분 동안 캐시 데이터를 fresh 상태로 유지
});
```

- `5분` 동안 `posts` 데이터를 신선하다고 간주한다.

- 5분이 지나면 `stale` 상태가 되어 `refetch`가 가능해진다.

## `gcTime` (가비지 컬렉션 시간)

- 설명: `gcTime`은 캐시 데이터가 "메모리에서 완전히 삭제되는 시간"을 의미한다.

- 기본값: `5분`

- 동작 방식:

  - `gcTime`이 지나기 전에는 **데이터가 메모리에 남아 있음** (stale 상태라도 메모리에 유지됨)

  - `gcTime`이 지나면 **해당 데이터가 캐시에서 삭제됨**

  - 삭제되면 다음 `useQuery`에서 다시 `fetch`할 때 **새로운 데이터를 요청**해야 함.

### 예제

```tsx
useQuery({
  queryKey: ["posts"],
  queryFn: fetchPosts,
  gcTime: 1000 * 60 * 10, // 10분 후 메모리에서 제거됨
});
```

- 10분 동안 캐시에 남아 있다가, 이후 메모리에서 삭제됨.

- 삭제된 후 `useQuery`가 실행되면 다시 데이터를 `fetch`해야 함

## `staleTime` vs `gcTime` 차이점

| 개념        | 설명                                       | 기본값 |
| ----------- | ------------------------------------------ | ------ |
| `staleTime` | 데이터가 "신선(fresh)"하다고 간주되는 시간 | 0ms    |
| `gcTime`    | 데이터가 캐시에서 완전히 삭제되는 시간     | `5분`  |

- 핵심 차이점

  - `staleTime`은 **데이터를 refetch할 수 있는 시점**을 결정.

  - `gcTime`은 **데이터가 캐시에서 사라지는 시점**을 결정.

- 흐름 예제 (`staleTime: 5분, gcTime: 10분`)

  1. 데이터를 가져오고 **5분 동안 fresh 상태 유지**

  2. 5분이 지나면 stale 상태가 되어 **refetch 가능**

  3. 아무도 사용하지 않는다면 10분 후 **캐시에서 삭제됨**

## 언제 사용하면 좋을까?

✅ staleTime을 길게 설정할 때
실시간 업데이트가 필요하지 않고, 캐시된 데이터를 오랫동안 재사용하고 싶을 때

> 예: 뉴스, 게시글 목록, 유저 프로필 등 자주 변하지 않는 데이터

✅ gcTime을 길게 설정할 때
앱에서 같은 데이터를 자주 요청하는 경우, 캐시가 빨리 삭제되면 불필요한 네트워크 요청이 증가할 수 있음

| 따라서 gcTime을 길게 설정하면 데이터가 메모리에 오래 유지되어 성능 최적화 가능

### 최적 설정 예제

```tsx
useQuery({
  queryKey: ["user"],
  queryFn: fetchUser,
  staleTime: 1000 * 60 * 5, // 5분 동안 fresh 상태 유지
  gcTime: 1000 * 60 * 30, // 30분 동안 메모리에 유지
});
```

- 5분 동안 fresh 상태 → refetch 안 함

- 5분 후 stale 상태 → refetch 가능

- 30분 후 gcTime이 지나면 캐시에서 삭제
