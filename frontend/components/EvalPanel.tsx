import { STAGE_LABELS, computeGoldMetrics, deriveChunkRows } from "@/lib/evalPanel";
import type { Turn } from "@/lib/types";

function PanelHeader({ question, onHide }: { question: string | undefined; onHide: () => void }) {
  return (
    <div className="sticky top-0 px-[18px] pt-4 pb-[14px] bg-surface">
      <div className="flex items-center justify-between">
        <span className="text-[13.5px] font-semibold text-text">Evaluation</span>
        <button onClick={onHide} className="text-[12.5px] text-accent-signal">
          Hide
        </button>
      </div>
      <p className="mt-1 text-[11.5px] text-text-dim truncate">
        {question ?? "Ask a question to see its evaluation trace."}
      </p>
    </div>
  );
}

export function EvalPanel({
  turn,
  stage,
  openRow,
  onStageChange,
  onToggleRow,
  reviewerVerdict,
  onReviewerVerdict,
  onHide,
}: {
  turn: Turn | undefined;
  stage: number;
  openRow: string | null;
  onStageChange: (stage: number) => void;
  onToggleRow: (rank: number) => void;
  reviewerVerdict: "correct" | "wrong" | undefined;
  onReviewerVerdict: (v: "correct" | "wrong") => void;
  onHide: () => void;
}) {
  if (!turn) {
    return (
      <aside className="w-[404px] shrink-0 border-l border-border bg-surface overflow-y-auto">
        <PanelHeader question={undefined} onHide={onHide} />
      </aside>
    );
  }

  const rows = deriveChunkRows(turn, stage);
  const { goldFound, goldRank } = computeGoldMetrics(turn, stage);
  const grounded = turn.groundedness !== null ? turn.groundedness.toFixed(2) : "—";
  const showingDense = stage < 1;
  const helper =
    reviewerVerdict === undefined
      ? "Your verdict is logged against this question's gold set."
      : `Saved to the review log for ${turn.question}.`;

  return (
    <aside className="w-[404px] shrink-0 border-l border-border bg-surface overflow-y-auto">
      <PanelHeader question={turn.question} onHide={onHide} />

      <div className="flex gap-[6px] px-[18px] pb-4">
        {STAGE_LABELS.map((label, i) => (
          <button
            key={label}
            onClick={() => onStageChange(i)}
            className={`text-[12px] px-[10px] py-1 rounded-[4px] border transition-colors ${
              stage === i
                ? "border-accent-signal bg-accent-signal text-bg font-medium"
                : "border-border bg-bg text-text-dim hover:text-text hover:border-accent-signal"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mx-[18px] mb-4 grid grid-cols-3 border border-border rounded-[5px] bg-bg overflow-hidden">
        <div className="px-3 py-[11px] border-r border-border">
          <div className="text-[11.5px] text-text-dim">Gold found</div>
          <div className="mt-1 font-mono text-[17px] text-accent-gold">{goldFound}</div>
        </div>
        <div className="px-3 py-[11px] border-r border-border">
          <div className="text-[11.5px] text-text-dim">Gold rank</div>
          <div className="mt-1 font-mono text-[17px] text-accent-gold">{goldRank}</div>
        </div>
        <div className="px-3 py-[11px]">
          <div className="text-[11.5px] text-text-dim">Grounded</div>
          <div className="mt-1 font-mono text-[17px] text-success">{grounded}</div>
        </div>
      </div>

      <div className="px-[18px] pb-2 flex items-center justify-between gap-2">
        <span className="text-[12.5px] text-text-dim">
          {showingDense ? "Dense candidates · top 50" : "Retrieved chunks · after rerank"}
        </span>
        <span className="text-[11.5px] text-text-dim shrink-0">Click a row for the chunk text</span>
      </div>

      <div>
        {rows.map((row) => {
          const rowKey = `${turn.id}:${row.rank}`;
          const isOpen = openRow === rowKey;
          const dim = row.note === "Not cited";
          return (
            <div key={row.chunkId}>
              <button
                type="button"
                onClick={() => onToggleRow(row.rank)}
                className={`w-full text-left flex items-center gap-3 px-[18px] py-[11px] border-t border-t-border border-l-2 ${
                  row.gold ? "border-l-accent-gold" : "border-l-border"
                } ${isOpen ? "bg-row-highlight" : "bg-bg"}`}
              >
                <span
                  className={`font-mono text-[12px] w-[22px] shrink-0 ${
                    row.gold ? "text-accent-gold" : "text-text-dim"
                  }`}
                >
                  {row.rank}
                </span>
                <span className="flex-1 min-w-0">
                  <span
                    className={`block font-mono text-[11.5px] truncate ${dim ? "text-text-dim" : "text-text"}`}
                  >
                    {row.path}
                  </span>
                  <span className="block text-[11.5px] text-text-dim">{row.note}</span>
                </span>
                <span className={`font-mono text-[12px] shrink-0 ${dim ? "text-text-dim" : "text-text"}`}>
                  {row.score}
                </span>
              </button>
              {isOpen && (
                <div className="px-[18px] pt-3 pb-[14px] pl-[22px] border-t border-border bg-bg text-[12.5px] leading-[1.6] text-text-dim italic">
                  {row.text}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="px-[18px] pt-4 pb-[22px] border-t border-border">
        <div className="text-[12.5px] text-text-dim mb-2">Reviewer verdict</div>
        <div className="flex gap-2">
          <button
            onClick={() => onReviewerVerdict("correct")}
            className={`flex-1 text-[13px] py-2 rounded-[4px] border ${
              reviewerVerdict === "correct"
                ? "bg-success border-success text-bg"
                : "border-border bg-bg text-success"
            }`}
          >
            Correct
          </button>
          <button
            onClick={() => onReviewerVerdict("wrong")}
            className={`flex-1 text-[13px] py-2 rounded-[4px] border ${
              reviewerVerdict === "wrong"
                ? "bg-danger border-danger text-bg"
                : "border-border bg-bg text-danger"
            }`}
          >
            Wrong
          </button>
        </div>
        <p className="mt-[10px] text-[12px] leading-[1.55] text-text-dim">{helper}</p>
      </div>
    </aside>
  );
}
