---
layout: page
title: Home
permalink: /
---

안녕하세요. **최명제**입니다.  
(여기에 한 줄 포지셔닝: 예. On-device / Agentic AI / Android platform 등)

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
