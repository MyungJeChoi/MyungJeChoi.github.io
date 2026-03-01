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
  <picture>
    <source
      type="image/webp"
      srcset="
        {{ '/assets/images/profile-440.webp' | relative_url }} 440w,
        {{ '/assets/images/profile-880.webp' | relative_url }} 880w
      "
      sizes="(max-width: 768px) 160px, 220px"
    />
    <img
      class="hero__image"
      srcset="
        {{ '/assets/images/profile-440.jpg' | relative_url }} 440w,
        {{ '/assets/images/profile-880.jpg' | relative_url }} 880w,
        {{ '/assets/images/profile.jpg' | relative_url }} 2304w
      "
      sizes="(max-width: 768px) 160px, 220px"
      src="{{ '/assets/images/profile-880.jpg' | relative_url }}"
      alt="최명제"
    />
  </picture>
  <div class="hero__content">
    <p>안녕하세요. <strong>최명제</strong>입니다.</p>
    <p>CV link: <a href="https://drive.google.com/file/d/14qXEVtlkrHqi7qTAGqEa5_LFJAyW6cIe/view?usp=sharing">here</a></p>
    <h3>Current Status:</h3>
    <ul>
      <li>Postdoctoral Researcher</li>
      <li>Numerical Computing &amp; Image Analysis Lab</li>
      <li>Research Institute of Mathematics</li>
      <li>Seoul National University</li>
    </ul>
    <h3>Research Interest:</h3>
    <ul>
      <li>Time Series Forecasting &amp; Anomaly Detection</li>
      <li>Graph Neural Networks</li>
      <li>Global Search Algorithms for GraphRAG</li>
      <li>LLM-enhanced NL2SQL Framework</li>
    </ul>
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
