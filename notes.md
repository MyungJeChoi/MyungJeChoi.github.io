---
layout: page
title: Notes
permalink: /notes/
---

{% assign items = site.notes | sort: "date" | reverse %}
{% for r in items %}
### [{{ r.title }}]({{ r.url | relative_url }})
{{ r.summary }}

{% if r.venue %}**Venue:** {{ r.venue }}{% endif %}{% if r.year %} ({{ r.year }}){% endif %}  
{% if r.paper %}[Paper]({{ r.paper }}){% endif %}{% if r.code %} · [Code]({{ r.code }}){% endif %}

---
{% endfor %}
