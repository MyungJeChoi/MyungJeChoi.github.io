---
layout: page
title: Projects
permalink: /projects/
---

## Heuristic NL2SQL

[Heuristic NL2SQL 프로젝트 모음]({% link projects_heuristic_nl2sql.md %})  

{% assign items = site.projects | where_exp: "p", "p.categories contains 'heuristic_nl2sql'" | sort: "date" | reverse %}
{% for p in items %}
### [{{ p.title }}]({{ p.url | relative_url }})
{{ p.summary }}

{% if p.tags %}**Tags:** {{ p.tags | join: ", " }}{% endif %}  
{% if p.github %}[GitHub]({{ p.github }}){% endif %}{% if p.demo %} · [Demo]({{ p.demo }}){% endif %}

---
{% endfor %}
