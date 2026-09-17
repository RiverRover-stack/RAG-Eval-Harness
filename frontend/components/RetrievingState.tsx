const STAGE_LABELS = [
  "Retrieving — embed…",
  "Retrieving — dense 50…",
  "Retrieving — rerank 8…",
  "Retrieving — answer…",
];

const SKELETON_WIDTHS = ["96%", "88%", "62%"];

export function RetrievingState({ stageIndex }: { stageIndex: number }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <span className="w-[7px] h-[7px] rounded-full bg-accent-signal" />
        <span className="text-[13px] text-text-dim">
          {STAGE_LABELS[stageIndex % STAGE_LABELS.length]}
        </span>
      </div>
      <div className="flex flex-col gap-[10px]">
        {SKELETON_WIDTHS.map((width, i) => (
          <div
            key={i}
            className="h-[13px] rounded-[3px] bg-surface animate-pulse-skeleton"
            style={{ width }}
          />
        ))}
      </div>
    </div>
  );
}
