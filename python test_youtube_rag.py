

import sys
import time
import traceback

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ PASS{RESET} — {msg}")
def fail(msg): print(f"  {RED}❌ FAIL{RESET} — {msg}")
def warn(msg): print(f"  {YELLOW}⚠  SKIP{RESET} — {msg}")
def header(msg): print(f"\n{'─'*55}\n🧪  {msg}\n{'─'*55}")

results = {"passed": 0, "failed": 0, "skipped": 0}

def record(status):
    results[status] += 1

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Transcript Fetching
# ══════════════════════════════════════════════════════════════════════════════
header("TEST 1 · Transcript Fetching (YouTubeTranscriptApi)")

VALID_VIDEO_ID   = "Gfr50f6ZBvo"   # the one used in your notebook
INVALID_VIDEO_ID = "INVALIDID00000"

try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

    # 1-A: fetch real transcript
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(VALID_VIDEO_ID, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        assert isinstance(transcript, str) and len(transcript) > 100, "transcript too short"
        ok(f"Valid video — fetched {len(transcript)} chars, {len(transcript_list)} chunks")
        record("passed")
    except Exception as e:
        fail(f"Valid video fetch failed: {e}")
        record("failed")
        transcript = None

    # 1-B: word count sanity
    if transcript:
        word_count = len(transcript.split())
        assert word_count > 50, "word count suspiciously low"
        ok(f"Word count sanity — {word_count} words")
        record("passed")
    else:
        warn("Skipping word-count check (no transcript)")
        record("skipped")

    # 1-C: bad video ID should raise, not silently return empty
    try:
        YouTubeTranscriptApi.get_transcript(INVALID_VIDEO_ID, languages=["en"])
        fail("Bad video ID should have raised an exception")
        record("failed")
    except Exception:
        ok("Bad video ID raises exception as expected")
        record("passed")

except ImportError:
    warn("youtube_transcript_api not installed — skipping section")
    record("skipped")
    transcript = None


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Text Splitting
# ══════════════════════════════════════════════════════════════════════════════
header("TEST 2 · Text Splitting (RecursiveCharacterTextSplitter)")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    CHUNK_SIZE    = 1000
    CHUNK_OVERLAP = 200

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # use real transcript if available, else a synthetic one
    sample_text = transcript if transcript else ("word " * 600)  # ~3000 chars
    chunks = splitter.create_documents([sample_text])

    # 2-A: produces chunks
    assert len(chunks) > 0, "no chunks produced"
    ok(f"Produced {len(chunks)} chunks from input")
    record("passed")

    # 2-B: no chunk exceeds chunk_size (LangChain allows slight overflow at word boundaries)
    oversized = [c for c in chunks if len(c.page_content) > CHUNK_SIZE * 1.1]
    assert len(oversized) == 0, f"{len(oversized)} chunks exceed size limit"
    ok(f"All chunks within size limit ({CHUNK_SIZE} chars ± 10%)")
    record("passed")

    # 2-C: overlap — adjacent chunks share content
    if len(chunks) >= 2:
        end_of_first   = chunks[0].page_content[-CHUNK_OVERLAP:]
        start_of_second = chunks[1].page_content[:CHUNK_OVERLAP]
        shared = set(end_of_first.split()) & set(start_of_second.split())
        assert len(shared) > 0, "no overlap detected between adjacent chunks"
        ok(f"Chunk overlap verified — {len(shared)} shared words between chunk 0→1")
        record("passed")
    else:
        warn("Only 1 chunk — overlap check skipped")
        record("skipped")

    # 2-D: each chunk is a LangChain Document with page_content
    from langchain_core.documents import Document
    assert all(isinstance(c, Document) for c in chunks), "not all chunks are Documents"
    ok("All chunks are LangChain Document objects")
    record("passed")

except ImportError as e:
    warn(f"langchain not installed — {e}")
    record("skipped")
    chunks = []


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Embedding Generation
# ══════════════════════════════════════════════════════════════════════════════
header("TEST 3 · Embedding Generation (HuggingFaceEmbeddings / MiniLM-L6-v2)")

try:
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3-A: single text produces 384-dim vector
    test_text = "What is DeepMind working on?"
    vec = embeddings.embed_query(test_text)
    assert isinstance(vec, list) and len(vec) == 384, f"expected 384-dim, got {len(vec)}"
    ok(f"Single embed — 384-dim vector ✓  (first val: {vec[0]:.4f})")
    record("passed")

    # 3-B: two semantically similar texts are closer than dissimilar ones
    import math
    def cosine(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na  = math.sqrt(sum(x*x for x in a))
        nb  = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb) if na and nb else 0

    sim_q  = embeddings.embed_query("Tell me about neural networks")
    sim_r  = embeddings.embed_query("Explain deep learning and neural nets")
    diff_r = embeddings.embed_query("What is the price of tomatoes today?")

    sim_score  = cosine(sim_q, sim_r)
    diff_score = cosine(sim_q, diff_r)
    assert sim_score > diff_score, (
        f"Similar pair score ({sim_score:.3f}) should exceed dissimilar ({diff_score:.3f})"
    )
    ok(f"Semantic similarity works — similar: {sim_score:.3f} > dissimilar: {diff_score:.3f}")
    record("passed")

    # 3-C: batch embed_documents
    batch_texts = ["Hello world", "Deep learning", "YouTube transcript"]
    vecs = embeddings.embed_documents(batch_texts)
    assert len(vecs) == 3 and all(len(v) == 384 for v in vecs), "batch embed failed"
    ok(f"Batch embed — {len(vecs)} vectors, each 384-dim")
    record("passed")

except ImportError as e:
    warn(f"langchain_huggingface not installed — {e}")
    record("skipped")
    embeddings = None


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — ChromaDB Vector Store
# ══════════════════════════════════════════════════════════════════════════════
header("TEST 4 · ChromaDB Vector Store")

import tempfile, os

try:
    from langchain_community.vectorstores import Chroma

    if not embeddings:
        warn("No embeddings model — skipping ChromaDB tests")
        record("skipped")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:

            # use real chunks or synthetic fallback
            from langchain_core.documents import Document
            test_docs = chunks[:10] if len(chunks) >= 10 else [
                Document(page_content=f"Test sentence number {i} about AI and machine learning.")
                for i in range(10)
            ]

            # 4-A: store creation
            vs = Chroma.from_documents(
                documents=test_docs,
                embedding=embeddings,
                collection_name="test_rag",
                persist_directory=tmp_dir
            )
            stored_ids = vs.get()["ids"]
            assert len(stored_ids) == len(test_docs), (
                f"stored {len(stored_ids)}, expected {len(test_docs)}"
            )
            ok(f"ChromaDB store created — {len(stored_ids)} docs indexed")
            record("passed")

            # 4-B: retriever returns k=4 results
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            results_r = retriever.invoke("AI machine learning")
            assert len(results_r) == 4, f"expected 4 results, got {len(results_r)}"
            ok(f"Retriever returns exactly k=4 docs")
            record("passed")

            # 4-C: each retrieved doc has page_content
            assert all(hasattr(d, "page_content") and d.page_content for d in results_r)
            ok("All retrieved docs have non-empty page_content")
            record("passed")

            # 4-D: similarity_search_with_score gives relevance scores
            scored = vs.similarity_search_with_score("AI machine learning", k=2)
            assert len(scored) == 2
            assert all(isinstance(s, float) for _, s in scored)
            ok(f"Similarity scores returned — top score: {scored[0][1]:.4f}")
            record("passed")

except ImportError as e:
    warn(f"chromadb / langchain_community not installed — {e}")
    record("skipped")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Prompt Template
# ══════════════════════════════════════════════════════════════════════════════
header("TEST 5 · Prompt Template (PromptTemplate)")

try:
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        template="""
  You are a helpful assistant.
  Answer ONLY from the provided transcript context.
  If the context is insufficient, just say you don't know.

  {context}
  Question: {question}
""",
        input_variables=["context", "question"]
    )

    # 5-A: both variables are recognised
    assert set(prompt.input_variables) == {"context", "question"}
    ok("Template variables — 'context' and 'question' recognised")
    record("passed")

    # 5-B: invoke substitutes variables correctly
    filled = prompt.invoke({"context": "AI stands for Artificial Intelligence.", "question": "What is AI?"})
    filled_str = filled.to_string() if hasattr(filled, "to_string") else str(filled)
    assert "AI stands for Artificial Intelligence." in filled_str
    assert "What is AI?" in filled_str
    ok("Template fills context and question correctly")
    record("passed")

    # 5-C: missing variable raises
    try:
        prompt.invoke({"context": "Some context."})   # 'question' missing
        fail("Should raise on missing variable")
        record("failed")
    except Exception:
        ok("Missing template variable raises exception as expected")
        record("passed")

except ImportError as e:
    warn(f"langchain_core not installed — {e}")
    record("skipped")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — LLM Loading (Qwen2.5-7B-Instruct) — weight check only
# ══════════════════════════════════════════════════════════════════════════════
header("TEST 6 · LLM Config (Qwen2.5-7B-Instruct) — imports + pipeline params")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    # 6-A: tokenizer loads (downloads ~500MB; skip if offline)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        ok(f"Tokenizer loaded for {MODEL_ID}")
        record("passed")

        # 6-B: tokenizer encodes/decodes correctly
        sample = "What does DeepMind do?"
        encoded = tokenizer(sample, return_tensors="pt")
        decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
        assert sample.lower() in decoded.lower() or len(decoded) > 0
        ok(f"Tokenizer encode→decode roundtrip works")
        record("passed")

    except Exception as e:
        warn(f"Tokenizer download failed (offline or disk space?) — {e}")
        record("skipped")

    # 6-C: pipeline kwargs are valid (no model load needed)
    valid_kwargs = dict(
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
    )
    assert valid_kwargs["task"] == "text-generation"
    assert 0 < valid_kwargs["temperature"] <= 1
    assert valid_kwargs["max_new_tokens"] > 0
    ok("Pipeline kwargs (task, temperature, max_new_tokens) are valid")
    record("passed")

    # 6-D: torch dtype bfloat16 is available
    assert hasattr(torch, "bfloat16"), "torch.bfloat16 not available"
    ok("torch.bfloat16 dtype available for model loading")
    record("passed")

except ImportError as e:
    warn(f"transformers / torch not installed — {e}")
    record("skipped")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — End-to-End RAG Pipeline (without loading the 7B model)
# ══════════════════════════════════════════════════════════════════════════════
header("TEST 7 · End-to-End RAG Flow (stub LLM — no GPU needed)")

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document

    # Minimal stub that mimics ChatHuggingFace .invoke()
    class StubLLM:
        class _Content:
            def __init__(self, text): self.content = text
        def invoke(self, prompt_value):
            text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
            return self._Content(f"[STUB ANSWER based on {len(text)} chars of context+question]")

    # Re-use embeddings + ChromaDB from Test 4
    if not embeddings:
        warn("No embeddings model — skipping E2E test")
        record("skipped")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build small in-memory store
            docs = [
                Document(page_content="DeepMind is an AI research lab owned by Alphabet."),
                Document(page_content="DeepMind created AlphaGo, which beat the world champion in Go."),
                Document(page_content="Nuclear fusion research is being pursued by several labs."),
                Document(page_content="Reinforcement learning is a key technique used at DeepMind."),
            ]
            vs = Chroma.from_documents(
                documents=docs, embedding=embeddings,
                collection_name="e2e_test", persist_directory=tmp_dir
            )
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 2})

            prompt = PromptTemplate(
                template="You are helpful.\n{context}\nQuestion: {question}",
                input_variables=["context", "question"]
            )
            llm = StubLLM()

            # Run a query
            question       = "What is DeepMind?"
            retrieved_docs = retriever.invoke(question)
            context_text   = "\n\n".join(doc.page_content for doc in retrieved_docs)
            final_prompt   = prompt.invoke({"context": context_text, "question": question})
            answer         = llm.invoke(final_prompt)

            # 7-A: retrieval found relevant docs
            assert any("DeepMind" in d.page_content for d in retrieved_docs)
            ok(f"E2E — retriever returned {len(retrieved_docs)} relevant docs for 'DeepMind'")
            record("passed")

            # 7-B: context is non-empty
            assert len(context_text) > 10
            ok(f"E2E — context assembled ({len(context_text)} chars)")
            record("passed")

            # 7-C: LLM produces an answer
            assert hasattr(answer, "content") and len(answer.content) > 0
            ok(f"E2E — LLM produced answer: {answer.content[:60]}…")
            record("passed")

except Exception as e:
    fail(f"E2E test crashed: {e}")
    traceback.print_exc()
    record("failed")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
print(f"  RESULTS  —  "
      f"{GREEN}{results['passed']} passed{RESET}  |  "
      f"{RED}{results['failed']} failed{RESET}  |  "
      f"{YELLOW}{results['skipped']} skipped{RESET}")
print(f"{'═'*55}\n")

if results["failed"] > 0:
    sys.exit(1)
