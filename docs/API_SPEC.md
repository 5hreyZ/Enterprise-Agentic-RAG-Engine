# Enterprise Agentic RAG Engine: REST & Streaming API Specification

## Base URL
```
http://localhost:8000
```

---

## 1. System Health & Telemetry

### `GET /health`
Returns system status and connectivity to Qdrant vector database and vLLM inference backend.

#### Response (`200 OK`):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "qdrant_connected": true,
  "vllm_connected": true
}
```

### `GET /metrics`
Returns real-time telemetry including indexed document counts, vector dimensions, and agent threshold configurations.

---

## 2. Ingestion API

### `POST /api/v1/documents/ingest`
Ingests, chunks, and indexes enterprise documents into both Qdrant vector storage and the BM25 sparse index.

#### Request Body:
```json
{
  "documents": [
    {
      "content": "Enterprise cloud security mandates AES-256 encryption at rest and TLS 1.3 in transit.",
      "metadata": {
        "source": "cloud_security_policy.md",
        "category": "security"
      },
      "doc_id": "sec_policy_01"
    }
  ],
  "chunk_size": 512,
  "chunk_overlap": 64
}
```

#### Response (`200 OK`):
```json
{
  "status": "success",
  "documents_processed": 1,
  "chunks_created": 3,
  "chunks_indexed": 3
}
```

---

## 3. Query Execution API

### `POST /api/v1/query`
Executes end-to-end multi-hop agentic RAG query with reasoning trace and citation attribution.

#### Request Body:
```json
{
  "query": "What are the SLA penalty credit formulas if uptime drops below 99%, and what is the Sev-1 response time?",
  "max_hops": 3
}
```

#### Response (`200 OK`):
```json
{
  "query": "What are the SLA penalty credit formulas if uptime drops below 99%, and what is the Sev-1 response time?",
  "answer": "Based on the verified enterprise documentation:\n- Tier-1 (Sev-1) incidents mandate an initial response time of < 15 minutes [Doc ID: sla_chunk_02].\n- If monthly availability drops below 99.00%, a 50% SLA service credit is applied [Doc ID: sla_chunk_03].",
  "is_multi_hop": true,
  "sub_queries": [
    "SLA availability penalty thresholds below 99%",
    "Severity-1 incident initial response time requirements"
  ],
  "citations": [
    {
      "chunk_id": "sla_chunk_02",
      "doc_id": "sla_policy",
      "source": "microservices_sla.md",
      "score": 0.8921,
      "text_preview": "Tier-1 (Critical / Severity-1): Initial Response Time: < 15 minutes..."
    }
  ],
  "faithfulness_score": 0.94,
  "verification_verdict": "PASS",
  "latency_ms": 342.15,
  "reasoning_trace": [
    {
      "step": "query_routing",
      "hop": 0,
      "details": { "is_multi_hop": true, "sub_queries": [...] }
    },
    {
      "step": "retrieval_hop_1",
      "hop": 1,
      "details": { "retrieved_count": 10, "relevant_count": 2 }
    }
  ]
}
```

---

## 4. Real-Time Streaming API

### `POST /api/v1/query/stream`
Streams answer tokens via Server-Sent Events (SSE).

#### Example cURL:
```bash
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare SOC 2 Type II audit log retention with financial audit guidelines."}'
```

#### Stream Output:
```
data: {"token": "Based "}
data: {"token": "on "}
data: {"token": "the "}
data: {"token": "verified "}
...
data: {"event": "complete", "faithfulness": 0.94, "verdict": "PASS", "latency_ms": 380.2}
```
