import { useState } from "react";
import { Shell, type View } from "./components/Shell";
import { Overview } from "./views/Overview";
import { Incidents } from "./views/Incidents";
import { Governance } from "./views/Governance";
import { AgentActivity } from "./views/AgentActivity";
import { System } from "./views/System";
import Dashboard from "./pages/Dashboard";
import AuditLog from "./pages/AuditLog";
import Product from "./pages/Product";
import Docs from "./pages/Docs";
import "./styles/view.css";

type Page = "console" | "dashboard" | "audit-log" | "product" | "docs";

export default function App() {
  const [page, setPage] = useState<Page>("console");
  const [view, setView] = useState<View>("overview");
  const [focusRun, setFocusRun] = useState<string | null>(null);

  // Standalone pages render outside the console Shell
  if (page === "dashboard") return <Dashboard />;
  if (page === "audit-log") return <AuditLog />;
  if (page === "product")   return <Product />;
  if (page === "docs")      return <Docs />;

  return (
    <Shell view={view} onView={setView}>
      <div key={view} className="view-enter">
        {view === "overview" && <Overview onView={setView} />}
        {view === "incidents" && <Incidents onView={setView} onFocusRun={setFocusRun} />}
        {view === "governance" && <Governance />}
        {view === "agent-activity" && <AgentActivity focusRun={focusRun} setFocusRun={setFocusRun} />}
        {view === "settings" && <System />}
      </div>
    </Shell>
  );
}
