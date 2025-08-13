# app.py — Redis Web RAG (cache-first) with Streamlit
# - Embeddings: SentenceTransformers ('msmarco-distilbert-base-v4', 768-dim)
# - Storage: RedisJSON (embeddings saved as JSON float arrays — decode_responses=True is OK)
# - Search: RediSearch KNN (COSINE)
# - Web fetch: googlesearch-python + httpx + trafilatura

import os, re, time, numpy as np
import streamlit as st
from dotenv import load_dotenv
import redis
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import httpx, trafilatura
from lxml import html as lxml_html
from googlesearch import search
from langdetect import detect_langs
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from openai import AzureOpenAI

# ---------- env & clients ----------
load_dotenv()

# Redis
r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    username=os.getenv("REDIS_USERNAME") or None,
    password=os.getenv("REDIS_PASSWORD") or None,
    ssl=bool(int(os.getenv("REDIS_SSL", "0"))),
    decode_responses=True  # OK because we store embeddings as JSON float arrays
)

INDEX_NAME = "websearch_cache_idx"
EMBED_DIM = 768

# Azure OpenAI
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
openai_key = os.getenv("AZURE_OPENAI_KEY", "")
client_o3 = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=openai_endpoint,
    api_key=openai_key
)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer('msmarco-distilbert-base-v4')

embedder = get_embedder()

# ---------- ensure index (idempotent) ----------
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.field import TextField, TagField, NumericField, VectorField

def ensure_index():
    try:
        r.ft(INDEX_NAME).info()
    except Exception:
        schema = (
            TextField("$.query",   as_name="query",   no_stem=True, sortable=True),
            TextField("$.title",   as_name="title"),
            TextField("$.snippet", as_name="snippet"),
            TagField("$.domain",   as_name="domain"),
            TagField("$.lang",     as_name="lang"),
            NumericField("$.fetched_at", as_name="fetched_at", sortable=True),
            TextField("$.metadata", as_name="metadata"),
            VectorField("$.embedding", "HNSW", {
                "TYPE": "FLOAT32", "DIM": EMBED_DIM, "DISTANCE_METRIC": "COSINE",
                "M": 16, "EF_CONSTRUCTION": 200
            }, as_name="embedding"),
        )
        r.ft(INDEX_NAME).create_index(
            schema,
            definition=IndexDefinition(prefix=["search_cache:"], index_type=IndexType.JSON)
        )

ensure_index()

# ---------- search utils (key-free Google + robust extraction) ----------
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
_TRACKING_PARAMS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "utm_id","gclid","fbclid","mc_cid","mc_eid","igshid","ref","mkt_tok"
}

def _domain(u:str)->str:
    return re.sub(r"^https?://(www\.)?([^/:]+).*$", r"\2", u)

def _html_title(s:str)->str:
    try:
        root = lxml_html.fromstring(s)
        t = (root.xpath('string(//title)') or '').strip() or (root.xpath('string(//h1[1])') or '').strip()
        if not t:
            metas = root.xpath('//meta[@property="og:title"]/@content | //meta[@name="twitter:title"]/@content')
            t = (metas[0] if metas else '').strip()
        return re.sub(r"\s+"," ",t)
    except Exception:
        return ""

def _strip_tags(s:str)->str:
    try:
        root = lxml_html.fromstring(s)
        return "\n".join(t.strip() for t in root.xpath("//text()") if t.strip())
    except Exception:
        return re.sub(r"<[^>]+>", " ", s)

def _canonical_from_html(s:str)->str:
    try:
        root = lxml_html.fromstring(s)
        hrefs = root.xpath('//link[translate(@rel,"CANONICAL","canonical")="canonical"]/@href')
        return hrefs[0].strip() if hrefs else ""
    except Exception:
        return ""

def _normalize_url(u:str, html_str:str="")->str:
    canon = _canonical_from_html(html_str)
    if canon.startswith("http"): u = canon
    parts = urlsplit(u); scheme = parts.scheme or "http"
    netloc = parts.netloc.lower()
    if netloc.endswith(":80"): netloc = netloc[:-3]
    if netloc.endswith(":443"): netloc = netloc[:-4]
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    qpairs = [(k,v) for (k,v) in parse_qsl(parts.query, keep_blank_values=True) if k not in _TRACKING_PARAMS]
    query = urlencode(qpairs, doseq=True)
    if path != "/" and path.endswith("/"): path = path[:-1]
    return urlunsplit((scheme, netloc, path, query, ""))

def _extract(html: str, url: str):
    txt = trafilatura.extract(html, url=url, include_comments=False, include_tables=False, favor_recall=True)
    if not txt: txt = _strip_tags(html) or ""
    title = ""
    for line in (txt.splitlines()[:10] if txt else []):
        if line.lower().startswith("title:"):
            title = line.split(":",1)[1].strip(); break
    if not title: title = _html_title(html) or re.sub(r'[?#].*$','',url).rstrip('/').rsplit('/',1)[-1].replace('-',' ').replace('_',' ')
    return {"title": title, "text": (txt or "").strip()}

def _looks_gated(text: str, html: str) -> bool:
    blob = f"{text}\n{html}".lower()
    bad = ["access denied","requires authorization","sign in","login","subscribe to read","subscription required","paywall","403 forbidden","captcha","are you a robot","request blocked","not authorized"]
    return any(m in blob for m in bad) or len(text.strip()) < 900

def web_search_google(query: str, k: int = 5, timeout: int = 15):
    urls = list(search(query, num_results=max(k*5, k)))
    pages = []
    with httpx.Client(follow_redirects=True, headers={"User-Agent": UA}) as client:
        for u in urls[:max(k*5, k)]:
            try:
                r0 = client.get(u, timeout=timeout); r0.raise_for_status()
                pages.append((u, r0.text))
            except Exception:
                pages.append((u, ""))

    now = int(time.time())
    items = []
    for u, h in pages:
        ex = _extract(h,u); canon = _normalize_url(u,h); text = ex["text"]
        if _looks_gated(text,h): continue
        lang = ""
        try:
            langs = detect_langs(text[:4000])
            if langs:
                best = max(langs, key=lambda l: l.prob)
                lang = best.lang if best.prob >= 0.6 else ""
        except Exception:
            pass
        items.append({
            "query": query, "title": ex["title"], "snippet": text[:4000].strip(),
            "domain": _domain(canon), "lang": lang, "fetched_at": now,
            "metadata": f"source=google;url={canon}",
        })
    # de-dup by url/domain, prefer longer snippet
    seen_url, seen_dom, out = set(), set(), []
    for it in sorted(items, key=lambda x: len(x["snippet"]), reverse=True):
        url = re.search(r"url=(\S+)", it["metadata"]).group(1) if "url=" in it["metadata"] else ""
        if url and url in seen_url: continue
        if it["domain"] in seen_dom: continue
        seen_url.add(url); seen_dom.add(it["domain"]); out.append(it)
    return out[:k]

# ---------- vector & Redis helpers ----------
def embed_text(text: str) -> bytes:
    v = embedder.encode(text, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32).tobytes()

def _knn(vec_bytes: bytes, k: int = 10):
    q = (Query(f'(*)=>[KNN {k} @embedding $v AS score]')
         .sort_by("score")
         .return_fields("score","title","snippet","domain","lang","fetched_at","metadata")
         .dialect(2))
    return r.ft(INDEX_NAME).search(q, query_params={"v": vec_bytes})

def _count_hits(vec_bytes: bytes, k: int = 50, max_cosine_dist: float = 0.55) -> int:
    res = _knn(vec_bytes, k=max(k,50))
    return sum(1 for d in res.docs if float(d.score) <= max_cosine_dist)

def _top_context(vec_bytes: bytes, k: int = 5, max_cosine_dist: float = 0.55):
    res = _knn(vec_bytes, k=max(k,50))
    filtered = [d for d in res.docs if float(d.score) <= max_cosine_dist][:k]
    if not filtered: filtered = res.docs[:k]
    items = []
    for d in filtered:
        m = re.search(r"url=(\S+)", getattr(d, "metadata", "") or "")
        items.append({
            "title": getattr(d,"title",""),
            "domain": getattr(d,"domain",""),
            "url": m.group(1) if m else "",
            "snippet": getattr(d,"snippet","")
        })
    return items

def save_docs_with_embeddings(docs, search_term: str):
    pipe = r.pipeline(transaction=False)
    for i, src in enumerate(docs, 1):
        text_for_embedding = f"{src.get('title','')}\n\n{src.get('snippet','')}".strip()
        vec = embedder.encode(text_for_embedding, normalize_embeddings=True).astype(np.float32)
        payload = {
            "query":      src.get("query") or search_term,
            "title":      src.get("title",""),
            "snippet":    (src.get("snippet","")[:4000]).strip(),
            "domain":     src.get("domain",""),
            "lang":       src.get("lang",""),
            "fetched_at": int(src.get("fetched_at") or time.time()),
            "metadata":   src.get("metadata",""),
            "embedding":  vec.tolist(),  # JSON-safe array of floats
        }
        key = f"search_cache:{payload['fetched_at']}:{i}"
        pipe.json().set(key, "$", payload)
    pipe.execute()

def llm_rewrite(query: str) -> str:
    try:
        resp = client_o3.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role":"system","content":
                 "Rewrite the user's request as one concise, high-recall web search query. No commentary."},
                {"role":"user","content":query}
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out.splitlines()[0] if out else query
    except Exception:
        return query

def answer_with_cache_or_search(question: str, k: int = 5, threshold: float = 0.55):
    search_term = llm_rewrite(question)
    qvec = embed_text(search_term)
    if _count_hits(qvec, k=50, max_cosine_dist=threshold) >= 3:
        ctx = _top_context(qvec, k=k, max_cosine_dist=threshold)
    else:
        fresh = web_search_google(search_term, k=k)
        if fresh:
            save_docs_with_embeddings(fresh, search_term)
        ctx = _top_context(qvec, k=k, max_cosine_dist=threshold)

    system_prompt = "You are a precise research assistant. Use ONLY the provided context. Cite domains inline. Be concise."
    ctx_txt = "\n\n".join(f"- {it['title']} ({it['domain']})\n  {it['url']}\n  {it['snippet'][:800]}" for it in ctx)
    user_msg = f"Question: {question}\nSearch term used: {search_term}\n\nContext:\n{ctx_txt}"
    resp = client_o3.chat.completions.create(
        model="o3-mini",
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_msg}],
    )
    return resp.choices[0].message.content, ctx

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Redis Web RAG", page_icon="🔎", layout="wide")
st.title("🔎 Redis Web RAG (cache-first)")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K context", 3, 10, 5)
    threshold = st.slider("Cosine distance threshold", 0.10, 0.90, 0.55, 0.01)
    st.caption("Lower distance = closer. ~0.5–0.6 is a reasonable band for SBERT.")
    st.caption("Env needed: REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY")

question = st.text_input("Ask a question to search + RAG:")
if st.button("Search"):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Searching / retrieving..."):
            answer, ctx = answer_with_cache_or_search(question.strip(), k=k, threshold=threshold)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Context")
        for i, c in enumerate(ctx, 1):
            st.markdown(f"**{i}. {c['title']}**  \n*{c['domain']}*  \n{c['url']}")
            st.caption(c["snippet"][:500] + ("..." if len(c["snippet"])>500 else ""))
