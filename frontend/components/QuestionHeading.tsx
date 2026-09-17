import { renderInlineCode } from "@/lib/inline";

export function QuestionHeading({ text }: { text: string }) {
  return (
    <h1
      className="text-[23px] font-semibold leading-[1.38] tracking-[-0.012em] text-text mb-[22px]"
      style={{ textWrap: "pretty" }}
    >
      {renderInlineCode(text, "font-mono text-[20px] text-accent-signal")}
    </h1>
  );
}
