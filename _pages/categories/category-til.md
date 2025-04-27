---
title: "TIL"
layout: archive
permalink: /categories/til/
author_profile: true
sidebar_main: true
---

<h2>📚 TIL 서브카테고리</h2>

{% assign til_posts = site.categories.til %}
{% assign til_subcategories = til_posts | map: "subcategory" | uniq | sort %}

{% for subcat in til_subcategories %}
{% if subcat %}
<h3 id="{{ subcat | slugify }}">{{ subcat | capitalize }}</h3>
<ul>
{% assign filtered_posts = til_posts | where: "subcategory", subcat %}
{% for post in filtered_posts %}
<li>
<a href="{{ post.url }}">{{ post.title }}</a> - <small>{{ post.date | date: "%Y-%m-%d" }}</small>
</li>
{% endfor %}
</ul>
{% endif %}
{% endfor %}
