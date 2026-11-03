"""Accessibility parser for source-preserving HTML publication."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_CONTEXTUAL_FULL_SIZE_LABEL_RE = re.compile(
    r"Open full[- ]size (?:Figure\s+\d+|figure),\s*(?P<context>.+)",
    re.IGNORECASE,
)


@dataclass
class _FigureAccessibilityFacts:
    """Markup facts collected for one rendered ``figure`` element."""

    figure_id: str
    has_image: bool = False
    has_caption: bool = False
    caption_count: int = 0
    caption_details_dom_order: list[str] = field(default_factory=list)
    image_sources: list[str] = field(default_factory=list)
    image_aria_details: list[str] = field(default_factory=list)
    details_count: int = 0
    details_id: str = ""
    details_classes: set[str] = field(default_factory=set)
    details_figure_label: str = ""
    details_expanded_initially: bool = False
    summary_count: int = 0
    description_paragraph_count: int = 0
    summary_chunks: list[str] = field(default_factory=list)
    description_chunks: list[str] = field(default_factory=list)
    details_open: bool = False
    summary_open: bool = False


def _is_contextual_full_size_label(value: str) -> bool:
    match = _CONTEXTUAL_FULL_SIZE_LABEL_RE.fullmatch(value)
    return bool(match and re.search(r"\w", match.group("context"), re.UNICODE))


class _AccessibilityParser(HTMLParser):
    """Collect bounded structural accessibility facts from one HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_languages: list[str] = []
        self.titles: list[str] = []
        self._title_chunks: list[str] | None = None
        self.main_count = 0
        self.main_content_id_count = 0
        self.focusable_main_content_count = 0
        self.skip_link_count = 0
        self.first_interactive_seen = False
        self.first_interactive_is_skip = False
        self.missing_image_alt = 0
        self.unlabelled_full_size_links = 0
        self.generic_full_size_links = 0
        self.duplicate_ids: set[str] = set()
        self._ids: set[str] = set()
        self._figure_stack: list[_FigureAccessibilityFacts] = []
        self.figure_facts: list[_FigureAccessibilityFacts] = []
        self.figures_missing_captions = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        classes = set((attributes.get("class") or "").split())
        is_skip_link = tag == "a" and "skip-link" in classes and attributes.get("href") == "#main-content"
        tabindex = attributes.get("tabindex")
        try:
            has_sequential_tabindex = tabindex is not None and int(tabindex) >= 0
        except ValueError:
            has_sequential_tabindex = False
        is_interactive = (
            (tag == "a" and bool(attributes.get("href")))
            or tag in {"button", "embed", "iframe", "object", "select", "summary", "textarea"}
            or (tag == "input" and (attributes.get("type") or "text").lower() != "hidden")
            or (tag in {"audio", "video"} and "controls" in attributes)
            or has_sequential_tabindex
        )
        if is_interactive and not self.first_interactive_seen:
            self.first_interactive_seen = True
            self.first_interactive_is_skip = is_skip_link
        if element_id:
            if element_id in self._ids:
                self.duplicate_ids.add(element_id)
            self._ids.add(element_id)

        if tag == "html":
            self.html_languages.append((attributes.get("lang") or "").strip())
        elif tag == "title":
            self._title_chunks = []
        elif tag == "main":
            self.main_count += 1
            if element_id == "main-content":
                self.main_content_id_count += 1
                if attributes.get("tabindex") == "-1":
                    self.focusable_main_content_count += 1
        elif tag == "a":
            if is_skip_link:
                self.skip_link_count += 1
            if "figure-full-size-link" in classes:
                aria_label = (attributes.get("aria-label") or "").strip()
                if not aria_label:
                    self.unlabelled_full_size_links += 1
                elif not _is_contextual_full_size_label(aria_label):
                    self.generic_full_size_links += 1
        elif tag == "figure":
            self._figure_stack.append(_FigureAccessibilityFacts(figure_id=(element_id or "").strip()))
        elif tag == "img":
            if not (attributes.get("alt") or "").strip():
                self.missing_image_alt += 1
            if self._figure_stack:
                facts = self._figure_stack[-1]
                facts.has_image = True
                facts.image_sources.append((attributes.get("src") or "").strip())
                facts.image_aria_details.append((attributes.get("aria-details") or "").strip())
        elif tag == "figcaption" and self._figure_stack:
            facts = self._figure_stack[-1]
            facts.has_caption = True
            facts.caption_count += 1
            facts.caption_details_dom_order.append("figcaption")
        elif tag == "details" and self._figure_stack:
            facts = self._figure_stack[-1]
            facts.details_count += 1
            facts.caption_details_dom_order.append("details")
            facts.details_id = (element_id or "").strip()
            facts.details_classes = classes
            facts.details_figure_label = (attributes.get("data-figure-label") or "").strip()
            facts.details_expanded_initially = "open" in attributes
            facts.details_open = True
        elif tag == "summary" and self._figure_stack:
            facts = self._figure_stack[-1]
            if facts.details_open:
                facts.summary_count += 1
                facts.summary_open = True
        elif tag == "p" and self._figure_stack:
            facts = self._figure_stack[-1]
            if facts.details_open and not facts.summary_open:
                facts.description_paragraph_count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_chunks is not None:
            self.titles.append("".join(self._title_chunks).strip())
            self._title_chunks = None
        elif tag == "summary" and self._figure_stack:
            self._figure_stack[-1].summary_open = False
        elif tag == "details" and self._figure_stack:
            self._figure_stack[-1].details_open = False
        elif tag == "figure" and self._figure_stack:
            facts = self._figure_stack.pop()
            self.figure_facts.append(facts)
            if facts.has_image and not facts.has_caption:
                self.figures_missing_captions += 1

    def handle_data(self, data: str) -> None:
        if self._title_chunks is not None:
            self._title_chunks.append(data)
        if self._figure_stack:
            facts = self._figure_stack[-1]
            if facts.details_open:
                if facts.summary_open:
                    facts.summary_chunks.append(data)
                else:
                    facts.description_chunks.append(data)

    def close(self) -> None:
        super().close()
        while self._figure_stack:
            facts = self._figure_stack.pop()
            self.figure_facts.append(facts)
            if facts.has_image and not facts.has_caption:
                self.figures_missing_captions += 1
