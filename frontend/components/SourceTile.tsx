import type { CitationOut } from "@/lib/types";

// Gold vs ordinary styling per the agreed design brief -- `gold` is only
// ever true when the backend matched the question to a
// loaded eval item AND this chunk is in its resolved gold set (ask.py's
// `_gold_chunk_ids`), never a generic brand accent.
//
// Clicking opens the evaluation panel on the matching chunk row (PR2) --
// this used to open `citation.url` in a new tab (PR1 stopgap), replaced now
// that the panel exists to "inspect the chunk" in place.
export function SourceTile({ citation, onClick }: { citation: CitationOut; onClick?: () => void }) {
  const title = citation.gold
    ? "Ground-truth source — click to inspect the chunk"
    : "Retrieved source — click to inspect the chunk";
  const toneClasses = citation.gold
    ? "border-accent-gold text-accent-gold bg-[rgba(214,169,74,0.08)] hover:bg-[rgba(214,169,74,0.2)]"
    : "border-border text-accent-signal bg-surface hover:border-accent-signal";

  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`inline-flex align-[2px] ml-[6px] font-mono text-[10.5px] px-[7px] py-[2px] rounded-full border cursor-pointer transition-colors ${toneClasses}`}
    >
      {citation.path}
    </button>
  );
}
