---
layout: page
title: Myungje Choi
permalink: /
---

<style>
.hero {
  display: flex;
  align-items: center;
  gap: 2rem;
}
.hero__image {
  width: 220px;
  max-width: 40vw;
  border-radius: 999px;
  object-fit: cover;
}
.hero__content {
  min-width: 0;
}
@media (max-width: 768px) {
  .hero {
    flex-direction: column;
    align-items: flex-start;
  }
  .hero__image {
    width: 160px;
  }
}
</style>

<section class="hero">
  <img
    class="hero__image"
    src="/assets/images/profile.jpg"
    alt="최명제"
  />
  <div class="hero__content">

안녕하세요. **최명제**입니다. ([CV](https://drive.google.com/file/d/14qXEVtlkrHqi7qTAGqEa5_LFJAyW6cIe/view?usp=sharing))  

### Current Status:
- Postdoctoral Researcher
- Numerical Computing & Image Analysis Lab
- Research Institute of Mathematics
- Seoul National University

### Research Interest:
- Time Series Forecasting & Anomaly Detection
- Graph Neural Networks
- Global Search Algorithms for GraphRAG
- LLM-enhanced NL2SQL Framework

  </div>
</section>

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
