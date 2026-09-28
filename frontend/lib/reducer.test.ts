import { describe, expect, it } from "vitest";
import { askReducer, initialAskState } from "./reducer";
import type { SSEEvent } from "./types";

describe("askReducer", () => {
  it("reduces a full meta -> retrieval -> token* -> citation -> done sequence", () => {
    let state = initialAskState;
    state = askReducer(state, {
      type: "SUBMIT",
      id: "t1",
      question: "How does FastAPI validate bodies?",
    });

    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].status).toBe("pending");
    expect(state.activeId).toBe("t1");

    const events: SSEEvent[] = [
      { event: "meta", data: { request_id: "r1", config_hash: "abc", k: 8 } },
      { event: "retrieval", data: { candidates: [], dense_candidates: [], timings: {} } },
      { event: "token", data: { t: "FastAPI validates " } },
      { event: "token", data: { t: "request bodies " } },
      { event: "token", data: { t: "with Pydantic [1]." } },
      {
        event: "citation",
        data: { index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: false, char_start: 0, char_end: 3 },
      },
      {
        event: "done",
        data: {
          request_id: "r1",
          question: "How does FastAPI validate bodies?",
          answer: "FastAPI validates request bodies with Pydantic [1].",
          citations: [{ index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: false }],
          coverage: 1,
          groundedness: 0.9,
          abstained: false,
          usage: { prompt_tokens: 10, completion_tokens: 5, cost_usd: 0.001 },
          latency_ms: 120,
        },
      },
    ];

    let state_afterCitation: typeof state | null = null;
    for (const sseEvent of events) {
      state = askReducer(state, { type: "SSE", id: "t1", sseEvent });
      if (sseEvent.event === "citation") state_afterCitation = state;
    }

    // Tile data is already present before `done` arrives, and the turn is
    // still "streaming" at that point -- this is the whole point of the fix.
    expect(state_afterCitation!.turns[0].citations).toEqual([
      { index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: false },
    ]);
    expect(state_afterCitation!.turns[0].status).toBe("streaming");

    const turn = state.turns[0];
    expect(turn.requestId).toBe("r1");
    expect(turn.status).toBe("done");
    expect(turn.answer).toBe("FastAPI validates request bodies with Pydantic [1].");
    expect(turn.citations).toEqual([
      { index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: false },
    ]);
    expect(turn.abstained).toBe(false);
  });

  it("appends citation tile data as citation events stream in, without duplicating a repeated index", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q" });
    state = askReducer(state, {
      type: "SSE",
      id: "t1",
      sseEvent: {
        event: "citation",
        data: { index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: true, char_start: 0, char_end: 3 },
      },
    });
    expect(state.turns[0].citations).toEqual([
      { index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: true },
    ]);

    // Duplicate index -- should not produce a second entry.
    state = askReducer(state, {
      type: "SSE",
      id: "t1",
      sseEvent: {
        event: "citation",
        data: { index: 1, chunk_id: "a", url: "https://x/a", path: "a", gold: true, char_start: 0, char_end: 3 },
      },
    });
    expect(state.turns[0].citations).toHaveLength(1);

    // A different index -- should append.
    state = askReducer(state, {
      type: "SSE",
      id: "t1",
      sseEvent: {
        event: "citation",
        data: { index: 2, chunk_id: "b", url: "https://x/b", path: "b", gold: false, char_start: 4, char_end: 7 },
      },
    });
    expect(state.turns[0].citations).toHaveLength(2);
  });

  it("marks streaming status and accumulates streamed text as tokens arrive", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q" });
    state = askReducer(state, { type: "SSE", id: "t1", sseEvent: { event: "token", data: { t: "hello " } } });
    state = askReducer(state, { type: "SSE", id: "t1", sseEvent: { event: "token", data: { t: "world" } } });

    expect(state.turns[0].status).toBe("streaming");
    expect(state.turns[0].streamedText).toBe("hello world");
  });

  it("records an error on its own turn and leaves other turns untouched", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q1" });
    state = askReducer(state, { type: "SUBMIT", id: "t2", question: "q2" });
    state = askReducer(state, {
      type: "SSE",
      id: "t2",
      sseEvent: { event: "error", data: { detail: "boom" } },
    });

    expect(state.turns[0].status).toBe("pending");
    expect(state.turns[1].status).toBe("error");
    expect(state.turns[1].errorDetail).toBe("boom");
  });

  it("advances the pending stage index via the timer, but stops once streaming starts", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q" });
    state = askReducer(state, { type: "ADVANCE_STAGE", id: "t1" });
    expect(state.turns[0].stageIndex).toBe(1);

    state = askReducer(state, { type: "SSE", id: "t1", sseEvent: { event: "token", data: { t: "hi" } } });
    state = askReducer(state, { type: "ADVANCE_STAGE", id: "t1" });
    expect(state.turns[0].stageIndex).toBe(1); // no longer pending -- tick is a no-op
  });

  it("records end-user feedback keyed by turn id", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q" });
    state = askReducer(state, { type: "SET_VERDICT", id: "t1", verdict: "good" });

    expect(state.verdicts.t1).toBe("good");
  });

  it("stores the retrieval event's candidates and dense_candidates on the turn", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q" });
    state = askReducer(state, {
      type: "SSE",
      id: "t1",
      sseEvent: {
        event: "retrieval",
        data: {
          candidates: [
            {
              chunk_id: "a",
              url: "https://x/a",
              title: "t",
              source_type: "docs",
              content: "content",
              scores: { rerank: 0.9 },
              ranks: { rerank: 1 },
              stages: ["rerank"],
              gold: true,
            },
          ],
          dense_candidates: [
            {
              chunk_id: "a",
              url: "https://x/a",
              title: "t",
              source_type: "docs",
              content: "content",
              scores: { rrf: 0.1 },
              ranks: { fusion: 1 },
              stages: ["fusion"],
              gold: true,
            },
          ],
          timings: {},
        },
      },
    });

    expect(state.turns[0].candidates).toHaveLength(1);
    expect(state.turns[0].candidates[0].chunk_id).toBe("a");
    expect(state.turns[0].denseCandidates[0].scores.rrf).toBe(0.1);
  });

  it("defaults view to chat and stage to 1 (Rerank 8)", () => {
    expect(initialAskState.view).toBe("chat");
    expect(initialAskState.stage).toBe(1);
    expect(initialAskState.openRow).toBeNull();
  });

  it("SET_VIEW switches between chat and eval", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SET_VIEW", view: "eval" });
    expect(state.view).toBe("eval");
    state = askReducer(state, { type: "SET_VIEW", view: "chat" });
    expect(state.view).toBe("chat");
  });

  it("SET_STAGE updates the stage and closes any open row", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "OPEN_ROW", key: "t1:1" });
    expect(state.openRow).toBe("t1:1");

    state = askReducer(state, { type: "SET_STAGE", stage: 0 });
    expect(state.stage).toBe(0);
    expect(state.openRow).toBeNull();
  });

  it("TOGGLE_ROW opens a closed row and closes it again on a second toggle", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "TOGGLE_ROW", key: "t1:1" });
    expect(state.openRow).toBe("t1:1");
    state = askReducer(state, { type: "TOGGLE_ROW", key: "t1:1" });
    expect(state.openRow).toBeNull();
  });

  it("TOGGLE_ROW switches to a different row rather than closing it, when another row is open", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "TOGGLE_ROW", key: "t1:1" });
    state = askReducer(state, { type: "TOGGLE_ROW", key: "t1:2" });
    expect(state.openRow).toBe("t1:2");
  });

  it("records reviewer verdicts separately from end-user verdicts", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q" });
    state = askReducer(state, { type: "SET_VERDICT", id: "t1", verdict: "good" });
    state = askReducer(state, { type: "SET_REVIEWER_VERDICT", id: "t1", verdict: "wrong" });

    expect(state.verdicts.t1).toBe("good");
    expect(state.reviewerVerdicts.t1).toBe("wrong");
  });

  it("SET_ACTIVE moves the header's active question away from the newest turn", () => {
    let state = initialAskState;
    state = askReducer(state, { type: "SUBMIT", id: "t1", question: "q1" });
    state = askReducer(state, { type: "SUBMIT", id: "t2", question: "q2" });
    expect(state.activeId).toBe("t2");

    state = askReducer(state, { type: "SET_ACTIVE", id: "t1" });
    expect(state.activeId).toBe("t1");
  });
});
