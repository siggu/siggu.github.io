---
layout: single
title: "CommonJS와 ES Module의 차이점"
categories: [til]
# tags: ["머신러닝", "웹앱"]
subcategory: JavaScript
---

> 2025-04-05 TIL-CommonJS와 ES Module의 차이점

# CommonJS와 ES Module의 차이점

CommonJS와 ES Module(ESM)은 자바스크립트에서 모듈을 관리하고 불러오는 두 가지 주요 방식이다.

## CommonJS (CJS)

Node.js에서 주로 사용되며, 모듈을 동기적으로 불러온다.

### **사용법**

```js
// 가져오기
const fs = require("fs");

// 내보내기
module.exports = { something };
```

### CommonJS의 **특징**

- **동기적 로딩**: 코드를 위에서 아래로 읽는 순서대로 실행되며, 동기적으로 모듈을 로딩함.

- **런타임에 로딩**: 실행 중에 `require()`를 호출해서 모듈을 불러올 수 있음.

- **Node.js 기본 모듈 시스템**

- `.cjs` 확장자 권장 (ESM과 혼용 시)

## ES Module (ESM)

자바스크립트 공식 표준 모듈 시스템으로, ECMAScript 2015(ES6)부터 도입되었다.

- **사용법**

  ```js
  // 가져오기
  import fs from "fs";

  // 내보내기
  export const something = "hello";
  export default myFunc;
  ```

- **특징**

  - **정적 분석 가능**: `import`/`export`는 **파일 최상단에서만** 사용되어야 함. 트리 쉐이킹(tree-shaking)에 유리함.

  - **비동기적 로딩 가능**: 브라우저에서 모듈을 비동기적으로 불러올 수 있음.

  - 브라우저와 Node.js 환경에서 모두 사용할 수 있다.

  - `.mjs` 확장자 또는 `"type": "module"` 설정 필요

## 주요 차이점

| 항목        | CommonJS (CJS)              | ES Module (ESM)                 |
| ----------- | --------------------------- | ------------------------------- |
| 문법        | `require`, `module.exports` | `import`, `export`              |
| 로딩 방식   | 동기                        | 비동기 가능                     |
| 실행 시점   | 런타임                      | 파싱 단계에서 미리 분석         |
| 트리 쉐이킹 | 불가능                      | 가능                            |
| 파일 확장자 | `.cjs`                      | `.mjs`, 또는 `"type": "module"` |
| 사용 환경   | Node.js 위주                | 브라우저 + Node.js              |
