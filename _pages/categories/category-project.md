---
title: "project"
layout: archive
permalink: categories/project
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.project %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
