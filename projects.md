---
layout: page
title: Projects
permalink: /projects/
---

{% assign items = site.projects | sort: "date" | reverse %}
{% for p in items %}
### [{{ p.title }}]({{ p.url | relative_url }})
{{ p.summary }}

{% if p.tags %}**Tags:** {{ p.tags | join: ", " }}{% endif %}  
{% if p.github %}[GitHub]({{ p.github }}){% endif %}{% if p.demo %} · [Demo]({{ p.demo }}){% endif %}

---
{% endfor %}
