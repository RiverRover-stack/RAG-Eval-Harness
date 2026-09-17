export function Header({ gatePassed }: { gatePassed: boolean }) {
  return (
    <header className="h-[56px] w-full shrink-0 bg-surface border-b border-border flex items-center px-5 gap-4">
      <div className="flex items-center gap-2">
        <span className="w-[9px] h-[9px] rounded-[2px] bg-accent-signal" />
        <span className="text-[14px] font-semibold text-text">FastAPI Docs Assistant</span>
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
