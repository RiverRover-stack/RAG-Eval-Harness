// The one piece of Phase 9's Ask flow with a unit test (docs/plan.md):
// reduces the `meta -> retrieval -> token* -> citation* -> done` SSE
// sequence from lib/sse.ts into UI state. Pure and side-effect free so it
// can be fed a canned event sequence in reducer.test.ts without a real
// fetch/stream.

import type { AskState, SSEEvent, Turn } from "./types";

export const initialAskState: AskState = {
  turns: [],
  activeId: null,
  draft: "",
  verdicts: {},
};

const STAGE_COUNT = 4; // embed / dense / rerank / answer

function newTurn(id: string, question: string): Turn {
  return {
    id,
    question,
    status: "pending",
    stageIndex: 0,
    requestId: null,
    streamedText: "",
    answer: "",
    citations: [],
    coverage: null,
    groundedness: null,
    abstained: false,
    errorDetail: null,
  };
}

function updateTurn(state: AskState, id: string, fn: (t: Turn) => Turn): AskState {
  return { ...state, turns: state.turns.map((t) => (t.id === id ? fn(t) : t)) };
}

export type AskAction =
  | { type: "SUBMIT"; id: string; question: string }
  | { type: "SSE"; id: string; sseEvent: SSEEvent }
  | { type: "ADVANCE_STAGE"; id: string }
  | { type: "SET_DRAFT"; text: string }
  | { type: "SET_VERDICT"; id: string; verdict: "good" | "bad" }
  | { type: "SET_ACTIVE"; id: string };

export function askReducer(state: AskState, action: AskAction): AskState {
  switch (action.type) {
    case "SUBMIT":
      return {
        ...state,
        turns: [...state.turns, newTurn(action.id, action.question)],
        activeId: action.id,
        draft: "",
      };

    case "SSE": {
      const { sseEvent } = action;
      switch (sseEvent.event) {
        case "meta":
          return updateTurn(state, action.id, (t) => ({ ...t, requestId: sseEvent.data.request_id }));
        case "retrieval":
        case "citation":
          // Trace-panel-only payloads (dense/rerank candidates, mid-stream
          // citation offsets) -- out of scope for the Answer-only chat mode.
          return state;
        case "token":
          return updateTurn(state, action.id, (t) => ({
            ...t,
            status: "streaming",
            streamedText: t.streamedText + sseEvent.data.t,
          }));
        case "done":
          return updateTurn(state, action.id, (t) => ({
            ...t,
            status: "done",
            requestId: sseEvent.data.request_id,
            answer: sseEvent.data.answer,
            citations: sseEvent.data.citations,
            coverage: sseEvent.data.coverage,
            groundedness: sseEvent.data.groundedness,
            abstained: sseEvent.data.abstained,
          }));
        case "error":
          return updateTurn(state, action.id, (t) => ({
            ...t,
            status: "error",
            errorDetail: sseEvent.data.detail,
          }));
        default:
          return state;
      }
    }

    case "ADVANCE_STAGE":
      // A no-op once real content has started arriving -- the 520ms timer
      // driving this keeps ticking in the background, but a turn past
      // "pending" no longer renders the RetrievingState it feeds.
      return updateTurn(state, action.id, (t) =>
        t.status === "pending" ? { ...t, stageIndex: (t.stageIndex + 1) % STAGE_COUNT } : t
      );

    case "SET_DRAFT":
      return { ...state, draft: action.text };

    case "SET_VERDICT":
      return { ...state, verdicts: { ...state.verdicts, [action.id]: action.verdict } };

    case "SET_ACTIVE":
      return { ...state, activeId: action.id };

    default:
      return state;
  }
}
