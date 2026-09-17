export function SuggestionChips({
  suggestions,
  onPick,
}: {
  suggestions: string[];
  onPick: (question: string) => void;
}) {
  if (suggestions.length === 0) return null;

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <span className="text-[12.5px] text-text-dim">Try</span>
      {suggestions.map((question) => (
        <button
          key={question}
          onClick={() => onPick(question)}
          className="text-[12.5px] px-[11px] py-[5px] rounded-full border border-border text-text-dim bg-surface hover:border-accent-signal hover:text-text"
        >
          {question}
        </button>
      ))}
    </div>
  );
}
