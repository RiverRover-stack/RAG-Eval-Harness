"use client";

import { useState } from "react";

export function AskInput({ onSubmit }: { onSubmit: (question: string) => void }) {
  const [value, setValue] = useState("");

  function submit() {
    const question = value.trim();
    if (!question) return;
    onSubmit(question);
    setValue("");
  }

  return (
    <div
      className="flex items-center gap-2 border border-border focus-within:border-accent-signal rounded-[6px] bg-surface"
      style={{ padding: "10px 12px 10px 15px" }}
    >
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") submit();
        }}
        placeholder="Ask a follow-up about FastAPI…"
        className="flex-1 bg-transparent outline-none text-[14.5px] text-text placeholder:text-text-dim"
      />
      <button
        onClick={submit}
        disabled={!value.trim()}
        className={`text-[13px] font-medium px-4 py-[7px] rounded-[4px] ${
          value.trim() ? "bg-accent-signal text-bg" : "bg-border text-text-dim"
        }`}
      >
        Ask
      </button>
    </div>
  );
}
