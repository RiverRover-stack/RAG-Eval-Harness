import { describe, expect, it } from "vitest";
import { computeGoldMetrics, deriveChunkRows, displayPath } from "./evalPanel";
import type { CandidateOut, Turn } from "./types";

function candidate(overrides: Partial<CandidateOut> = {}): CandidateOut {
  return {
    chunk_id: "a",
    url: "https://fastapi.tiangolo.com/tutorial/dependencies/#L61",
    title: "t",
    source_type: "docs",
    content: "chunk text",
    scores: {},
    ranks: {},
    stages: [],
    gold: false,
    ...overrides,
  };
}

function turn(overrides: Partial<Turn> = {}): Turn {
  return {
    id: "t1",
    question: "q",
    status: "done",
    stageIndex: 3,
    requestId: "r1",
    streamedText: "",
    answer: "answer [1].",
    citations: [],
    coverage: 1,
    groundedness: 0.9,
    abstained: false,
    errorDetail: null,
    candidates: [],
    denseCandidates: [],
    ...overrides,
  };
}

describe("displayPath", () => {
  it("strips scheme/host and leading/trailing slashes, keeping any fragment", () => {
    expect(displayPath("https://fastapi.tiangolo.com/tutorial/dependencies/#L61")).toBe(
      "tutorial/dependencies#L61"
    );
  });

  it("falls back to the raw url when it doesn't parse", () => {
    expect(displayPath("not-a-url")).toBe("not-a-url");
  });
});

describe("deriveChunkRows", () => {
  it("reads the dense pool for stage < 1 and the reranked list for stage >= 1", () => {
    const t = turn({
      candidates: [candidate({ chunk_id: "reranked" })],
      denseCandidates: [candidate({ chunk_id: "dense" })],
    });

    expect(deriveChunkRows(t, 0).map((r) => r.chunkId)).toEqual(["dense"]);
    expect(deriveChunkRows(t, 1).map((r) => r.chunkId)).toEqual(["reranked"]);
    expect(deriveChunkRows(t, 2).map((r) => r.chunkId)).toEqual(["reranked"]);
  });

  it("marks cited chunks by cross-referencing turn.citations, and notes gold+cited distinctly", () => {
    const t = turn({
      citations: [{ index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: true }],
      candidates: [
        candidate({ chunk_id: "a", gold: true }),
        candidate({ chunk_id: "b", gold: false }),
        candidate({ chunk_id: "c", gold: true }),
      ],
    });

    const rows = deriveChunkRows(t, 1);
    expect(rows[0]).toMatchObject({ chunkId: "a", cited: true, gold: true, note: "Ground truth · cited" });
    expect(rows[1]).toMatchObject({ chunkId: "b", cited: false, gold: false, note: "Not cited" });
    expect(rows[2]).toMatchObject({ chunkId: "c", cited: false, gold: true, note: "Not cited" });
  });

  it("scores the dense stage off rrf and rerank stages off rerank, falling back to rrf", () => {
    const t = turn({
      candidates: [candidate({ chunk_id: "a", scores: { rrf: 0.1, rerank: 0.8 } })],
      denseCandidates: [candidate({ chunk_id: "a", scores: { rrf: 0.1, rerank: 0.8 } })],
    });

    expect(deriveChunkRows(t, 0)[0].score).toBe("0.100");
    expect(deriveChunkRows(t, 1)[0].score).toBe("0.800");
  });

  it("falls back to rrf at the rerank stage when the reranker is disabled", () => {
    const t = turn({ candidates: [candidate({ chunk_id: "a", scores: { rrf: 0.42 } })] });
    expect(deriveChunkRows(t, 1)[0].score).toBe("0.420");
  });

  it("assigns rank as 1-based position in the active list", () => {
    const t = turn({ candidates: [candidate({ chunk_id: "a" }), candidate({ chunk_id: "b" })] });
    const rows = deriveChunkRows(t, 1);
    expect(rows.map((r) => r.rank)).toEqual([1, 2]);
  });
});

describe("computeGoldMetrics", () => {
  it("reports the lowest-ranked gold chunk's 1-based rank when found", () => {
    const t = turn({
      candidates: [
        candidate({ chunk_id: "a", gold: false }),
        candidate({ chunk_id: "b", gold: true }),
      ],
    });
    expect(computeGoldMetrics(t, 1)).toEqual({ goldFound: "Yes", goldRank: "2" });
  });

  it("never fabricates -- no gold chunk in the active list reads as em-dash, not 'No'", () => {
    const t = turn({ candidates: [candidate({ chunk_id: "a", gold: false })] });
    expect(computeGoldMetrics(t, 1)).toEqual({ goldFound: "—", goldRank: "—" });
  });

  it("is reactive to the active stage -- can differ between dense and reranked lists", () => {
    const t = turn({
      candidates: [candidate({ chunk_id: "a", gold: false })],
      denseCandidates: [
        candidate({ chunk_id: "a", gold: false }),
        candidate({ chunk_id: "gold-only-in-dense", gold: true }),
      ],
    });
    expect(computeGoldMetrics(t, 0)).toEqual({ goldFound: "Yes", goldRank: "2" });
    expect(computeGoldMetrics(t, 1)).toEqual({ goldFound: "—", goldRank: "—" });
  });
});
