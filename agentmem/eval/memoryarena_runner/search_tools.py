"""Lightweight search tools for MemoryArena Progressive Web Search."""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

@dataclass(frozen=True)
class SearchDocument:
    title: str
    url: str
    snippet: str

    def render(self, *, max_tokens: int = 512) -> str:
        text = f"Title: {self.title}\nURL: {self.url}\nSnippet: {self.snippet}"
        tokens = text.split()
        if len(tokens) > max_tokens:
            text = " ".join(tokens[:max_tokens])
        return text

def fetch_document_text(url: str, *, timeout: int = 10) -> str:
    """Fetch a search result page and return rough visible text.

    This intentionally stays dependency-free for remote experiment hosts. If a
    site blocks fetching, callers fall back to the search snippet.
    """
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 MemoryArenaHarness/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""
        body = response.read(200_000).decode("utf-8", errors="replace")
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    return _clean(body)

class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.docs: list[SearchDocument] = []
        self._in_title = False
        self._in_snippet = False
        self._href = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._pending_title: tuple[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            self._href = attr.get("href", "")
            self._title_parts = []
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            title = _clean(" ".join(self._title_parts))
            url = _unwrap_ddg_url(self._href)
            self._pending_title = (title, url)
            self._in_title = False
        elif self._in_snippet and tag in {"a", "div"}:
            snippet = _clean(" ".join(self._snippet_parts))
            if self._pending_title and snippet:
                title, url = self._pending_title
                if title and url:
                    self.docs.append(SearchDocument(title=title, url=url, snippet=snippet))
            self._pending_title = None
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)

def duckduckgo_search(query: str, *, top_k: int = 5, timeout: int = 15) -> list[SearchDocument]:
    """Return DuckDuckGo HTML search results without requiring an API key."""
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 MemoryArenaHarness/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    parser = _DuckDuckGoParser()
    parser.feed(body)
    return parser.docs[:top_k]

def render_search_observations(
    query: str,
    *,
    top_k: int = 5,
    truncate_tokens: int = 512,
) -> tuple[str, list[SearchDocument]]:
    docs = duckduckgo_search(query, top_k=top_k)
    if not docs:
        return "No search results returned.", []
    enriched: list[SearchDocument] = []
    for doc in docs:
        try:
            page_text = fetch_document_text(doc.url)
        except Exception:
            page_text = ""
        enriched.append(
            SearchDocument(
                title=doc.title,
                url=doc.url,
                snippet=page_text or doc.snippet,
            )
        )
    blocks = [
        f"[Search result {i}]\n{doc.render(max_tokens=truncate_tokens)}"
        for i, doc in enumerate(enriched, start=1)
    ]
    return "\n\n".join(blocks), enriched

def render_multi_search_observations(
    queries: list[str],
    *,
    top_k: int = 5,
    truncate_tokens: int = 512,
) -> tuple[str, list[SearchDocument]]:
    """Run multiple paper-style search calls and render fetched documents."""
    all_docs: list[SearchDocument] = []
    blocks: list[str] = []
    seen_urls: set[str] = set()
    for call_id, query in enumerate([q for q in queries if q.strip()], start=1):
        text, docs = render_search_observations(
            query,
            top_k=top_k,
            truncate_tokens=truncate_tokens,
        )
        kept_docs = []
        for doc in docs:
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)
            kept_docs.append(doc)
            all_docs.append(doc)
        if kept_docs:
            rendered = "\n\n".join(
                f"[Search call {call_id} result {i}]\n{doc.render(max_tokens=truncate_tokens)}"
                for i, doc in enumerate(kept_docs, start=1)
            )
        else:
            rendered = text
        blocks.append(f"Search call {call_id} query: {query}\n{rendered}")
    return "\n\n".join(blocks) if blocks else "No search queries issued.", all_docs

def _clean(text: str) -> str:
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def _unwrap_ddg_url(url: str) -> str:
    parsed = urllib.parse.urlparse(html.unescape(url))
    query = urllib.parse.parse_qs(parsed.query)
    uddg = query.get("uddg")
    if uddg:
        return uddg[0]
    return html.unescape(url)
