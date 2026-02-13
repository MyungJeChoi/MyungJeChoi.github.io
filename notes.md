---
layout: page
title: Notes
permalink: /notes/
---

{% assign items = site.notes | sort: "date" | reverse %}
{% for n in items %}
- {{ n.date | date: "%Y-%m-%d" }} · [{{ n.title }}]({{ n.url | relative_url }}){% if n.tags %} — {{ n.tags | join: ", " }}{% endif %}
{% endfor %}
