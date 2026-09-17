// Pure derivations for the evaluation panel (components/EvalPanel.tsx) --
// kept out of the reducer so the reducer stays about SSE-sequencing only,
// and so this is as easy to unit-test in isolation as reducer.test.ts does
// for the reducer itself.

import type { CandidateOut, Turn } from "./types";

// The design brief specs 4 pills ("Embed"/"Dense 50"/"Rerank 8"/"Answer"),
// with "Embed" and "Dense 50" both rendering the identical pre-rerank pool
// (there's no retrieval stage that produces embedding-only candidates
// distinct from the top-50 dense/fusion list) -- collapsed to one "Dense"
// pill per your call, since two controls that always show the same data
// regardless of config is more confusing than a labeling simplification.
export const STAGE_LABELS = ["Dense", "Rerank 8", "Answer"] as const;

// Stage index 0 ("Dense") renders the pre-rerank pool; 1/2 ("Rerank 8"/
// "Answer") render the final reranked list.
export function activeCandidates(turn: Turn, stage: number): CandidateOut[] {
  return stage < 1 ? turn.denseCandidates : turn.candidates;
}

// Mirrors ask.py's `_display_path()` -- short slug for a chunk's path, e.g.
// `tutorial/dependencies#L61` -- since `_candidate_out()` only sends `url`,
// not a precomputed path (candidate.gold/content were the only additive
// fields scoped for PR2; deriving path client-side avoids a third).
export function displayPath(url: string): string {
  try {
    const parsed = new URL(url);
    let path = parsed.pathname.replace(/^\/+|\/+$/g, "");
    if (parsed.hash) {
      const fragment = parsed.hash.slice(1);
      path = path ? `${path}#${fragment}` : fragment;
    }
    return path || url;
  } catch {
    return url;
  }
}

function noteFor(gold: boolean, cited: boolean): string {
  if (!cited) return "Not cited";
  return gold ? "Ground truth · cited" : "Cited";
}

// dense/embed stages score off the fusion (RRF) score -- there's no
// per-candidate "dense" score, fusion.py never separates it out. rerank/
// answer stages prefer the reranker's score, falling back to rrf for a
// reranker-disabled config (no "rerank" key on scores in that case).
function scoreFor(candidate: CandidateOut, stage: number): number | null {
  const score = stage < 1 ? candidate.scores.rrf : (candidate.scores.rerank ?? candidate.scores.rrf);
  return score ?? null;
}

export interface ChunkRow {
  rank: number;
  chunkId: string;
  path: string;
  cited: boolean;
  gold: boolean;
  note: string;
  score: string;
  text: string;
}

export function deriveChunkRows(turn: Turn, stage: number): ChunkRow[] {
  const citedIds = new Set(turn.citations.map((c) => c.chunk_id));
  return activeCandidates(turn, stage).map((candidate, i) => {
    const cited = citedIds.has(candidate.chunk_id);
    const score = scoreFor(candidate, stage);
    return {
      rank: i + 1,
      chunkId: candidate.chunk_id,
      path: displayPath(candidate.url),
      cited,
      gold: candidate.gold,
      note: noteFor(candidate.gold, cited),
      score: score === null ? "—" : score.toFixed(3),
      text: candidate.content,
    };
  });
}

export interface GoldMetrics {
  goldFound: string;
  goldRank: string;
}

// Scans the active (stage-selected) list for the lowest-ranked gold chunk.
// No separate backend metric, no eval-item lookup on the frontend: a
// question with no gold chunk anywhere in the active list reads "—" for
// both, whether that's because the question isn't a known eval-set item or
// because retrieval genuinely missed the gold chunk -- deliberately not
// disambiguated (see the PR2 brief), so this never fabricates a "found"
// that didn't happen.
export function computeGoldMetrics(turn: Turn, stage: number): GoldMetrics {
  const list = activeCandidates(turn, stage);
  const goldIndex = list.findIndex((c) => c.gold);
  if (goldIndex === -1) return { goldFound: "—", goldRank: "—" };
  return { goldFound: "Yes", goldRank: String(goldIndex + 1) };
}
