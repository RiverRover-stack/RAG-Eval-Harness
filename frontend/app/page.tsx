"use client";

import { useEffect, useReducer, useRef, useState } from "react";
import { Header } from "@/components/Header";
import { TurnView } from "@/components/Turn";
import { AskInput } from "@/components/AskInput";
import { SuggestionChips } from "@/components/SuggestionChips";
import { EvalPanel } from "@/components/EvalPanel";
import { askReducer, initialAskState } from "@/lib/reducer";
import { streamAsk } from "@/lib/sse";
import { API_BASE } from "@/lib/config";

export default function Home() {
  const [state, dispatch] = useReducer(askReducer, initialAskState);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const askedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    fetch(`${API_BASE}/api/suggestions?n=3`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setSuggestions)
      .catch(() => setSuggestions([]));
  }, []);

  async function ask(question: string) {
    const id = crypto.randomUUID();
    askedRef.current.add(question);
    dispatch({ type: "SUBMIT", id, question });

    const timer = setInterval(() => dispatch({ type: "ADVANCE_STAGE", id }), 520);
    try {
      for await (const sseEvent of streamAsk(question, API_BASE)) {
        dispatch({ type: "SSE", id, sseEvent });
        if (sseEvent.event !== "meta" && sseEvent.event !== "retrieval") {
          clearInterval(timer);
        }
      }
    } finally {
      clearInterval(timer);
    }
  }

  function submitFeedback(turnId: string, requestId: string | null, verdict: "good" | "bad") {
    dispatch({ type: "SET_VERDICT", id: turnId, verdict });
    dispatch({ type: "SET_ACTIVE", id: turnId });
    if (!requestId) return;
    fetch(`${API_BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId, verdict }),
    }).catch(() => {
      // best-effort -- feedback isn't load-bearing for the answer already shown
    });
  }

  function submitVerdict(turnId: string, requestId: string | null, verdict: "correct" | "wrong") {
    dispatch({ type: "SET_REVIEWER_VERDICT", id: turnId, verdict });
    if (!requestId) return;
    fetch(`${API_BASE}/api/verdict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId, verdict }),
    }).catch(() => {
      // best-effort, same as submitFeedback -- not load-bearing for the panel already shown
    });
  }

  function openEvalForCitation(turnId: string, chunkId: string) {
    dispatch({ type: "SET_ACTIVE", id: turnId });
    dispatch({ type: "SET_VIEW", view: "eval" });
    dispatch({ type: "SET_STAGE", stage: 1 }); // Rerank 8, also clears openRow
    const turn = state.turns.find((t) => t.id === turnId);
    const rank = turn?.candidates.findIndex((c) => c.chunk_id === chunkId) ?? -1;
    if (rank >= 0) dispatch({ type: "OPEN_ROW", key: `${turnId}:${rank + 1}` });
  }

  // The gate indicator (and, once built, latency) in the header always
  // describes the *active* question, not the newest one -- source-tile
  // clicks and Yes/No feedback can both move activeId away from the latest
  // turn (see openEvalForCitation/submitFeedback above).
  const activeTurn = state.turns.find((t) => t.id === state.activeId);
  const gatePassed = activeTurn?.status === "done" && activeTurn.abstained === false;
  const remainingSuggestions = suggestions.filter((q) => !askedRef.current.has(q));
  const panelOpen = state.view === "eval";

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <Header
        gatePassed={Boolean(gatePassed)}
        view={state.view}
        onViewChange={(view) => dispatch({ type: "SET_VIEW", view })}
      />
      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 overflow-y-auto">
          <div
            className={`mx-auto px-8 pt-7 pb-[26px] flex flex-col ${
              panelOpen ? "max-w-[760px] min-[1100px]:max-w-[860px]" : "max-w-[760px]"
            }`}
          >
            {state.turns.map((turn, i) => (
              <TurnView
                key={turn.id}
                turn={turn}
                verdict={state.verdicts[turn.id]}
                onVerdict={(v) => submitFeedback(turn.id, turn.requestId, v)}
                isLast={i === state.turns.length - 1}
                onCitationClick={(chunkId) => openEvalForCitation(turn.id, chunkId)}
                showEvalLink={!panelOpen}
                onShowEvaluation={() => {
                  dispatch({ type: "SET_ACTIVE", id: turn.id });
                  dispatch({ type: "SET_VIEW", view: "eval" });
                }}
              />
            ))}
            <div className={state.turns.length > 0 ? "mt-2" : ""}>
              <AskInput onSubmit={ask} />
              <SuggestionChips suggestions={remainingSuggestions} onPick={ask} />
            </div>
          </div>
        </main>
        {panelOpen && (
          <div className="hidden min-[1100px]:flex shrink-0">
            <EvalPanel
              turn={activeTurn}
              stage={state.stage}
              openRow={state.openRow}
              onStageChange={(stage) => dispatch({ type: "SET_STAGE", stage })}
              onToggleRow={(rank) =>
                activeTurn && dispatch({ type: "TOGGLE_ROW", key: `${activeTurn.id}:${rank}` })
              }
              reviewerVerdict={activeTurn ? state.reviewerVerdicts[activeTurn.id] : undefined}
              onReviewerVerdict={(v) =>
                activeTurn && submitVerdict(activeTurn.id, activeTurn.requestId, v)
              }
              onHide={() => dispatch({ type: "SET_VIEW", view: "chat" })}
            />
          </div>
        )}
      </div>
    </div>
  );
}
