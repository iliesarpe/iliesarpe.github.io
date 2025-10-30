---
layout: standalone
title: Research Projects
permalink: /projects/
---

# Research Projects

Here are some of my latest publications and ongoing projects:

<ul>
  {% for page in site.pages %}
    {% if page.path contains 'projects/' and page.title and page.url != '/projects/index.html' %}
      <li><a href="{{ page.url }}">{{ page.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

