import React from "react";
import axios from "axios";
import { PlayCircle, Loader2, AlertCircle } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

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
      const res = await axios.post(`${API}/run_campaign`, { objective });
      setResult(res.data);
    } catch (err) {
      setError("Failed to run campaign");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground" data-testid="campaign-runner-title">Campaign Runner</h1>
        <p className="text-muted-foreground mt-1">Start and observe AI-powered marketing campaigns</p>
      </div>
      <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
        <label className="block text-sm font-medium text-foreground mb-2">Campaign Objective</label>
        <textarea
          data-testid="campaign-objective-input"
          value={objective}
          onChange={e => setObjective(e.target.value)}
          placeholder="e.g., Launch a new product targeting millennials..."
          className="w-full h-32 px-4 py-3 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
        />
        {error && (
          <div className="mt-3 flex items-center space-x-2 text-red-500 text-sm" data-testid="campaign-error-message">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}
        <button
          onClick={handleRun}
          disabled={isRunning}
          data-testid="run-campaign-button"
          className="mt-4 w-full bg-primary hover:bg-primary/90 text-primary-foreground font-medium py-3 px-6 rounded-lg transition-all disabled:opacity-50 flex items-center justify-center space-x-2 shadow-lg shadow-primary/20"
        >
          {isRunning ? <><Loader2 size={20} className="animate-spin" /><span>Running...</span></> : <><PlayCircle size={20} /><span>Run Campaign</span></>}
        </button>
      </div>
      {result && (
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-foreground mb-4">Campaign Result</h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Campaign ID:</span>
              <span className="text-foreground font-mono" data-testid="campaign-id">{result.campaign_id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Status:</span>
              <span className="text-foreground font-medium" data-testid="campaign-status-indicator">{result.status}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Current Action:</span>
              <span className="text-foreground" data-testid="current-action">{result.current_action}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CampaignRunner;
