import React from "react";
import axios from "axios";
import { PlayCircle, Loader2, AlertCircle } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const CampaignRunner = () => {
  const [objective, setObjective] = React.useState("");
  const [isRunning, setIsRunning] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [result, setResult] = React.useState(null);

  const handleRun = async () => {
    if (!objective.trim()) {
      setError("Please enter a campaign objective");
      return;
    }

    setIsRunning(true);
    setError(null);

    try {
      // Prepare payload matching backend's RunCampaignRequest
      const payload = {
        state: {
          campaign_id: null,
          brief: objective,
          history: [],
          errors: []
        },
        execution_plan: ["PLAN"] // Action enum as string
      };

      const res = await axios.post(`${BACKEND_URL}/run_campaign`, payload);
      setResult(res.data);

    } catch (err) {
      console.error(err);
      setError("Failed to run campaign");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Campaign Runner</h1>
        <p className="text-muted-foreground mt-1">
          Start and observe AI-powered marketing campaigns
        </p>
      </div>

      <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
        <label className="block text-sm font-medium text-foreground mb-2">
          Campaign Objective
        </label>
        <textarea
          value={objective}
          onChange={e => setObjective(e.target.value)}
          placeholder="e.g., Launch a new product targeting millennials..."
          className="w-full h-32 px-4 py-3 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
        />

        {error && (
          <div className="mt-3 flex items-center space-x-2 text-red-500 text-sm">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <button
          onClick={handleRun}
          disabled={isRunning}
          className="mt-4 w-full bg-primary hover:bg-primary/90 text-primary-foreground font-medium py-3 px-6 rounded-lg transition-all disabled:opacity-50 flex items-center justify-center space-x-2 shadow-lg shadow-primary/20"
        >
          {isRunning ? (
            <>
              <Loader2 size={20} className="animate-spin" />
              <span>Running...</span>
            </>
          ) : (
            <>
              <PlayCircle size={20} />
              <span>Run Campaign</span>
            </>
          )}
        </button>
      </div>

      {result && (
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-foreground mb-4">Campaign Result</h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Campaign ID:</span>
              <span className="text-foreground font-mono">{result.campaign_id || "N/A"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Status:</span>
              <span className={`text-foreground font-medium ${result.errors?.length ? "text-red-500" : "text-green-500"}`}>
                {result.errors?.length ? "Failed" : "Success"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Current Action:</span>
              <span className="text-foreground">{result.current_action || "N/A"}</span>
            </div>
            {result.memory_context && result.memory_context.documents?.length > 0 && (
              <div className="mt-3">
                <span className="text-muted-foreground text-sm">Context from past campaigns:</span>
                <ul className="list-disc ml-5 text-sm">
                  {result.memory_context.documents.map((doc, idx) => (
                    <li key={idx}>{doc}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default CampaignRunner;
