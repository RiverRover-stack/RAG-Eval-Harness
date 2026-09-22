export function FeedbackRow({
  verdict,
  onVerdict,
  onShowEvaluation,
}: {
  verdict: "good" | "bad" | undefined;
  onVerdict: (v: "good" | "bad") => void;
  // Omitted (undefined) while the evaluation panel is already open --
  // the link only ever makes sense as a way to open it (design brief).
  onShowEvaluation?: () => void;
}) {
  const prompt =
    verdict === "good"
      ? "Marked helpful"
      : verdict === "bad"
        ? "Marked not helpful"
        : "Did this answer your question?";

  return (
    <div className="mt-6 flex items-center gap-[14px]">
      <span className="text-[13px] text-text-dim">{prompt}</span>
      <button
        onClick={() => onVerdict("good")}
        className={`text-[13px] px-[14px] py-[6px] rounded-[4px] border bg-surface ${
          verdict === "good" ? "border-success text-success" : "border-border text-text"
        }`}
      >
        Yes
      </button>
      <button
        onClick={() => onVerdict("bad")}
        className={`text-[13px] px-[14px] py-[6px] rounded-[4px] border bg-surface ${
          verdict === "bad" ? "border-danger text-danger" : "border-border text-text"
        }`}
      >
        No
      </button>
      {onShowEvaluation && (
        <button onClick={onShowEvaluation} className="ml-auto text-[12.5px] text-accent-signal">
          Show evaluation
        </button>
      )}
    </div>
  );
}
