// Mirrors src/rag_eval/api/routes/ask.py's response/SSE payload shapes --
// field names match exactly so no translation layer is needed.

export interface CitationOut {
  index: number;
  chunk_id: string;
  url: string;
  path: string;
  gold: boolean;
}

export interface UsageOut {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  cost_usd: number | null;
}

export interface DonePayload {
  request_id: string;
  question: string;
  answer: string;
  citations: CitationOut[];
  coverage: number;
  groundedness: number;
  abstained: boolean;
  usage: UsageOut;
  latency_ms: number;
}

export type SSEEvent =
  | { event: "meta"; data: { request_id: string; config_hash: string; k: number } }
  | { event: "retrieval"; data: { candidates: unknown[]; timings: Record<string, number> } }
  | { event: "token"; data: { t: string } }
  | {
      event: "citation";
      data: { index: number; chunk_id: string; url: string; char_start: number; char_end: number };
    }
  | { event: "done"; data: DonePayload }
  | { event: "error"; data: { detail: string } };

export type TurnStatus = "pending" | "streaming" | "done" | "error";

// `id` is a client-generated id (stable React key / reducer target),
// independent of `requestId` (the backend's request_id, known only once
// `meta` arrives, and what /api/feedback needs).
export interface Turn {
  id: string;
  question: string;
  status: TurnStatus;
  stageIndex: number; // 0..3, drives the "Retrieving" status line
  requestId: string | null;
  streamedText: string;
  answer: string;
  citations: CitationOut[];
  coverage: number | null;
  groundedness: number | null;
  abstained: boolean;
  errorDetail: string | null;
}

export interface AskState {
  turns: Turn[];
  activeId: string | null;
  draft: string;
  verdicts: Record<string, "good" | "bad">;
}
