---
layout: category
title: "TIL"
category: til
permalink: /til/
---

{% for post in site.categories.til %}

- [{{ post.title }}]({{ post.url }}) ({{ post.date | date: "%Y-%m-%d" }})
  {% endfor %}
