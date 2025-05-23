---
layout: single
title: "원숭이 시뮬레이터란?"
categories: [project]
subcategory: forge-simulator
# tags: ["머신러닝", "웹앱"]
---

> 2025-04-27 project-원숭이 시뮬레이터란?

## 이 글을 작성하게 된 계기

나는 [원숭이 시뮬레이터](https://forge-simulator.vercel.app) 페이지를 약 한 달*(2025-03-31 ~ 2025-04-27)*간 운영하였다. (현재 진행형)

이 페이지를 왜 운영하게 되었는지, 나의 각종 생각을 적어보고

궁극적으로 **구글 애널리틱스**를 이용해 사용자에 대한 각종 정보를 정리하고, 어떤 정보와 그에 따른 효과를 얻었는지 적어보는 글이다.

## 원숭이 시뮬레이터는 왜 생겼는가?

위 페이지는 마인크래프트 rpg 서버인 레오서버 내에 존재하는 무기 강화 및 열쇠 뽑기를 실제 확률에 기반하여 구현해 놓은 일종의 시뮬레이터이다.

[레오서버 공식 위키](https://reosv.gitbook.io/reo-server)에는 강화나 뽑기에 대한 확률이 공개되어 있지 않고, 디스코드 채널에 존재한다. 또한, 직접 게임에 들어가야만 확인할 수 있는 정보들이 존재한다.

무엇보다도, [메이플 주문서 시뮬레이터](https://gongnomok.com)처럼 무기 강화를 시뮬레이션 해보면 좋을 것 같다는 생각을 했다. 유저 중 아무도 유사한 사이트를 만든 사람이 없었기 때문이다.

사실은 이 사이트를 이용하지 않았으면 좋겠다는 마음으로 만든 사이트이다. 그 이유는 아래에서 설명할 수 있을 것 같다.

## 이름이 왜 원숭이 시뮬레이터 인데?

무기의 등급이 **일반, 고급, 희귀, 영웅, 전설 / 필멸** 등급으로 나뉜다.

일반부터 전설은 순서대로 성능 및 가격이 비싸고, 필멸은 얻기 더 힘들 뿐만 아니라 성능 또한 전설에는 조금 못 미치는 편이라고 생각한다.(개인적인 의견)

일정 스펙에 도달한 유저는 영웅 무기를 사용하다가 전설로 넘어가는 것이 일반적이다. 하지만, 전설 무기와 필멸 무기는 수요는 많지만 공급이 많지 않아 영웅 무기에 어쩔수 없이 정체하고 있는 사람이 많다.(필자도 그렇다.)

영웅 무기의 일정 강화 수준은 전설 무기의 성능을 뛰어넘기도 한다. 하지만, 그만큼 리스크가 존재하게 된다. 게다가, 이미 전설 무기를 보유한 사람도 일명 **엔드 스펙**에 도전하기 위해 낮은 확률에 도전하곤 한다. 이처럼 하이 리스크 하이 리턴에 도전하는 사람을 **원숭이**라고 부른다.

> 필자와 같이 영웅 무기들만 있어도 존재하는 모든 보스와 컨텐츠를 즐길 수 있는 스펙이긴 하지만, rpg의 특징상 스펙 경쟁이 일어날 수밖에 없는 구조이긴 한 것 같다.

주변에서 일명 **원숭이 강화**를 시도하는 사람이 늘어나면서 실패하는 사람은 물론 게임을 떠나는 사람도 보았다.

이를 보고 싶지 않았던 필자는 강화에 쉽게 도전하지 못하도록 실제 확률에 기반하여 시뮬레이터를 만들고 **여기에서도 강화를 실패하는데 실제로도 실패하지 않을까?** 라는 생각을 하도록 기대하였다.

~~(주변의 말을 들었을 때 강화를 누르지 않게 되는 사람도 있었지만 오히려 강화를 누르게 되는 사람이 더 존재했다고 한다..)~~

> 하루는 동접자 수가 300명 정도일 때 일일 최대 활성화 유저가 124명이었던 것을 보면 생각보다 사람들이 강화에 미쳐있는 느낌을 받았다...

## 이 글을 마치며

다음 글에는 웹 페이지를 운영하면서 **구글 애널리틱스**를 도입해 어떤 지표를 얻고, 어떤 효과를 보았는지 알아볼 것이다.
