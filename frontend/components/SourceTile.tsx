import type { CitationOut } from "@/lib/types";

// Gold vs ordinary styling per the agreed design brief -- `gold` is only
// ever true when the backend matched the question to a
// loaded eval item AND this chunk is in its resolved gold set (ask.py's
// `_gold_chunk_ids`), never a generic brand accent.
export function SourceTile({ citation }: { citation: CitationOut }) {
  const title = citation.gold
    ? "Ground-truth source — click to inspect the chunk"
    : "Retrieved source — click to inspect the chunk";
  const toneClasses = citation.gold
    ? "border-accent-gold text-accent-gold bg-[rgba(214,169,74,0.08)] hover:bg-[rgba(214,169,74,0.2)]"
    : "border-border text-accent-signal bg-surface hover:border-accent-signal";

  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noreferrer"
      title={title}
      className={`inline-flex align-[2px] ml-[6px] font-mono text-[10.5px] px-[7px] py-[2px] rounded-full border cursor-pointer transition-colors ${toneClasses}`}
    >
      {citation.path}
    </a>
  );
}
