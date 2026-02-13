---
layout: page
title: Myungje Choi
permalink: /
---

안녕하세요. **최명제**입니다.  \
Research Interest:
- Online Traffic Prediction
- GraphRAG Efficiency

## Featured Projects
{% assign items = site.projects | sort: "date" | reverse %}
{% for p in items limit: 3 %}
- [{{ p.title }}]({{ p.url | relative_url }}) — {{ p.summary }}
{% endfor %}

## Recent Researches
{% assign items = site.researches | sort: "date" | reverse %}
{% for r in items limit: 3 %}
- [{{ r.title }}]({{ r.url | relative_url }}) — {{ r.summary }}
{% endfor %}

## Recent Notes
{% assign items = site.notes | sort: "date" | reverse %}
{% for n in items limit: 5 %}
- [{{ n.title }}]({{ n.url | relative_url }})
{% endfor %}
