import type { AskState } from "@/lib/types";

const VIEW_OPTIONS: { value: AskState["view"]; label: string }[] = [
  { value: "chat", label: "Answer only" },
  { value: "eval", label: "Answer + evaluation" },
];

export function Header({
  gatePassed,
  view,
  onViewChange,
}: {
  gatePassed: boolean;
  view: AskState["view"];
  onViewChange: (view: AskState["view"]) => void;
}) {
  return (
    <header className="h-[56px] w-full shrink-0 bg-surface border-b border-border flex items-center px-5 gap-4">
      <div className="flex items-center gap-2">
        <span className="w-[9px] h-[9px] rounded-[2px] bg-accent-signal" />
        <span className="text-[14px] font-semibold text-text">FastAPI Docs Assistant</span>
      </div>
      <div
        className="flex items-center p-[2px] border border-border rounded-[5px] bg-bg"
        role="tablist"
      >
        {VIEW_OPTIONS.map((option) => (
          <button
            key={option.value}
            role="tab"
            aria-selected={view === option.value}
            onClick={() => onViewChange(option.value)}
            className={`text-[13px] px-[13px] py-[5px] rounded-[3px] ${
              view === option.value ? "bg-border text-text font-medium" : "text-text-dim font-normal"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
      <div className="flex-1" />
      {gatePassed && (
        <div className="flex items-center gap-2">
          <span className="w-[6px] h-[6px] rounded-full bg-success" />
          <span className="text-[13px] text-success">Gate passed</span>
        </div>
      )}
    </header>
  );
}
