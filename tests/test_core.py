import unittest

from contractlens.answer import NO_EVIDENCE, answer_question, evidence_context
from contractlens.core import Passage, VectorIndex, chunk_pages, extract_pdf_pages


class ToyEncoder:
    def encode(self, texts):
        return [[float("terminate" in t.lower()), float("payment" in t.lower())] for t in texts]


class CoreTests(unittest.TestCase):
    def test_chunks_keep_page_and_are_bounded(self):
        pages = [("a.pdf", 2, "Termination " * 80), ("a.pdf", 3, "Payment " * 80)]
        chunks = chunk_pages(pages, size=120, overlap=20)
        self.assertGreater(len(chunks), 2)
        self.assertEqual({chunk.page for chunk in chunks}, {2, 3})
        self.assertTrue(all(len(chunk.text) <= 120 for chunk in chunks))
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))

    def test_retrieval_filters_documents_and_cites_right_page(self):
        passages = [
            Passage("A.pdf", 4, "Terminate with notice", "a"),
            Passage("B.pdf", 7, "Payment is due", "b"),
        ]
        index = VectorIndex(passages, ToyEncoder())
        hits = index.search("termination", documents={"A.pdf"})
        self.assertEqual([(h.passage.document, h.passage.page) for h in hits], [("A.pdf", 4)])
        self.assertIn("[S1] A.pdf, page 4", evidence_context(hits))

    def test_no_llm_call_without_sources(self):
        self.assertEqual(answer_question("question", [], "", "model"), NO_EVIDENCE)

    def test_rejects_non_pdf(self):
        with self.assertRaisesRegex(ValueError, "not a PDF"):
            extract_pdf_pages("fake.pdf", b"plain text")


if __name__ == "__main__":
    unittest.main()
