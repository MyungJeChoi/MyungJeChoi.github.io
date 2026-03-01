---
layout: page
title: "Projects - LLM-augmented NL2SQL"
permalink: /projects/llm_nl2sql/
---

{% assign items = site.projects | where_exp: "p", "p.categories contains 'llm_nl2sql'" | sort: "date" | reverse %}

{% if items.size == 0 %}
## No posts

등록된 `LLM-augmented NL2SQL` 프로젝트가 없습니다.
{% else %}

{% for p in items %}
### [{{ p.title }}]({{ p.url | relative_url }})
{{ p.summary }}

{% if p.tags %}**Tags:** {{ p.tags | join: ", " }}{% endif %}  
{% if p.github %}[GitHub]({{ p.github }}){% endif %}{% if p.demo %} · [Demo]({{ p.demo }}){% endif %}

---
{% endfor %}
{% endif %}
