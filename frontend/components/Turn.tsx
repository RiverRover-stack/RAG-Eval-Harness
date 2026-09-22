import { QuestionHeading } from "./QuestionHeading";
import { AnswerBody } from "./AnswerBody";
import { RetrievingState } from "./RetrievingState";
import { FeedbackRow } from "./FeedbackRow";
import type { Turn as TurnState } from "@/lib/types";

export function TurnView({
  turn,
  verdict,
  onVerdict,
  isLast,
  onCitationClick,
  showEvalLink,
  onShowEvaluation,
}: {
  turn: TurnState;
  verdict: "good" | "bad" | undefined;
  onVerdict: (v: "good" | "bad") => void;
  isLast: boolean;
  onCitationClick: (chunkId: string) => void;
  showEvalLink: boolean;
  onShowEvaluation: () => void;
}) {
  return (
    <div className={isLast ? "" : "pb-[26px] mb-[26px] border-b border-border"}>
      <QuestionHeading text={turn.question} />
      {turn.status === "pending" && <RetrievingState stageIndex={turn.stageIndex} />}
      {turn.status === "error" && (
        <div className="border border-danger rounded-[6px] p-4 text-[14px] text-danger">
          {turn.errorDetail ?? "Something went wrong."}
        </div>
      )}
      {(turn.status === "streaming" || turn.status === "done") && (
        <>
          <AnswerBody
            answer={turn.status === "streaming" ? turn.streamedText : turn.answer}
            citations={turn.citations}
            onCitationClick={onCitationClick}
          />
          {turn.status === "done" && (
            <FeedbackRow
              verdict={verdict}
              onVerdict={onVerdict}
              onShowEvaluation={showEvalLink ? onShowEvaluation : undefined}
            />
          )}
        </>
      )}
    </div>
  );
}
