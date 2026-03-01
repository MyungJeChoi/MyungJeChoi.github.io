---
layout: page
title: "Projects - Heuristic NL2SQL"
permalink: /projects/heuristic_nl2sql/
---

{% assign items = site.projects | where_exp: "p", "p.categories contains 'heuristic_nl2sql'" | sort: "date" | reverse %}

{% if items.size == 0 %}
## No posts

등록된 `Heuristic NL2SQL` 프로젝트가 없습니다.
{% else %}

{% for p in items %}
### [{{ p.title }}]({{ p.url | relative_url }})
{{ p.summary }}

{% if p.github %}[GitHub]({{ p.github }}){% endif %}{% if p.demo %} · [Demo]({{ p.demo }}){% endif %}

---
{% endfor %}
{% endif %}
