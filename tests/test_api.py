import unittest
from enterprise_rag.api.routes import handle_query, handle_ingest, handle_stream_query

class TestAPIHandlers(unittest.TestCase):
    def test_ingest_and_query_flow(self):
        docs = [
            {
                "content": "Enterprise microservice API gateway rate limiting is capped at 5000 req/sec.",
                "metadata": {"doc": "gateway_policy"},
                "doc_id": "gw_01",
            }
        ]
        # Ingestion
        ingest_res = handle_ingest(docs)
        self.assertEqual(ingest_res["status"], "success")
        self.assertEqual(ingest_res["documents_processed"], 1)
        self.assertGreater(ingest_res["chunks_created"], 0)

        # Query
        query_res = handle_query("What is the rate limit for the API gateway?")
        self.assertIn("query", query_res)
        self.assertIn("answer", query_res)
        self.assertIn("citations", query_res)
        self.assertTrue(query_res["faithfulness_score"] >= 0.0)

    def test_streaming_query(self):
        stream_gen = handle_stream_query("API gateway rate limiting")
        tokens = list(stream_gen)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(any("data:" in t for t in tokens))

if __name__ == "__main__":
    unittest.main()
