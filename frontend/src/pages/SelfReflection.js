import React from "react";
import { Brain, AlertTriangle } from "lucide-react";

const SelfReflection = () => {
  const reflections = [
    {
      action: "CONTENT_GENERATION",
      attempt: 2,
      error: "Initial tone too formal, adjusted for casual audience",
      timestamp: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      action: "PERSONA_ANALYSIS",
      attempt: 1,
      error: "Demographics data incomplete, enriched with external sources",
      timestamp: new Date(Date.now() - 7200000).toISOString(),
    },
    {
      action: "RESEARCH",
      attempt: 3,
      error: "API rate limit reached, implemented exponential backoff",
      timestamp: new Date(Date.now() - 10800000).toISOString(),
    },
    {
      action: "EXPERIMENT_DESIGN",
      attempt: 2,
      error: "Control group size insufficient, adjusted sample distribution",
      timestamp: new Date(Date.now() - 14400000).toISOString(),
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <Brain size={32} className="text-primary" />
        <div>
          <h1 className="text-3xl font-bold text-foreground" data-testid="self-reflection-title">Self Reflection</h1>
          <p className="text-muted-foreground mt-1">Agent learning and retry visualization</p>
        </div>
      </div>

      <div className="bg-card rounded-lg border border-border shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full" data-testid="reflection-table">
            <thead className="bg-accent/50 border-b border-border">
              <tr>
                <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Action</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Attempt</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Error / Learning</th>
                <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {reflections.map((reflection, idx) => (
                <tr 
                  key={idx} 
                  className="hover:bg-accent/30 transition-colors"
                  data-testid={`reflection-row-${idx}`}
                >
                  <td className="px-6 py-4">
                    <span className="text-sm font-medium text-foreground">{reflection.action}</span>
                  </td>
                  <td className="px-6 py-4">
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary border border-primary/20">
                      Attempt {reflection.attempt}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-start space-x-2">
                      <AlertTriangle size={16} className="text-yellow-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-muted-foreground">{reflection.error}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="text-sm text-muted-foreground">
                      {new Date(reflection.timestamp).toLocaleString()}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <div className="text-2xl font-bold text-foreground" data-testid="total-retries">12</div>
          <div className="text-sm text-muted-foreground mt-1">Total Retries</div>
        </div>
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <div className="text-2xl font-bold text-foreground" data-testid="success-rate">87%</div>
          <div className="text-sm text-muted-foreground mt-1">Success Rate</div>
        </div>
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <div className="text-2xl font-bold text-foreground" data-testid="avg-attempts">1.8</div>
          <div className="text-sm text-muted-foreground mt-1">Avg Attempts</div>
        </div>
      </div>
    </div>
  );
};

export default SelfReflection;
