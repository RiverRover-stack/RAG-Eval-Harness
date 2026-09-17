"use client";

import { useEffect, useReducer, useRef, useState } from "react";
import { Header } from "@/components/Header";
import { TurnView } from "@/components/Turn";
import { AskInput } from "@/components/AskInput";
import { SuggestionChips } from "@/components/SuggestionChips";
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
    if (!requestId) return;
    fetch(`${API_BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId, verdict }),
    }).catch(() => {
      // best-effort -- feedback isn't load-bearing for the answer already shown
    });
  }

  const activeTurn = state.turns.at(-1);
  const gatePassed = activeTurn?.status === "done" && activeTurn.abstained === false;
  const remainingSuggestions = suggestions.filter((q) => !askedRef.current.has(q));

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <Header gatePassed={Boolean(gatePassed)} />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[760px] px-8 pt-7 pb-[26px] flex flex-col">
          {state.turns.map((turn, i) => (
            <TurnView
              key={turn.id}
              turn={turn}
              verdict={state.verdicts[turn.id]}
              onVerdict={(v) => submitFeedback(turn.id, turn.requestId, v)}
              isLast={i === state.turns.length - 1}
            />
          ))}
          <div className={state.turns.length > 0 ? "mt-2" : ""}>
            <AskInput onSubmit={ask} />
            <SuggestionChips suggestions={remainingSuggestions} onPick={ask} />
          </div>
        </div>
      </main>
    </div>
  );
}
