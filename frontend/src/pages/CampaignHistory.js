import React from "react";
import axios from "axios";
import { ChevronRight, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const CampaignHistory = () => {
  const [campaigns, setCampaigns] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const navigate = useNavigate();

  React.useEffect(() => {
    fetchCampaigns();
  }, []);

  const fetchCampaigns = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/campaigns`);
      setCampaigns(Array.isArray(response.data) ? response.data : []);
    } catch (error) {
      console.error("Failed to fetch campaigns:", error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 size={18} className="text-green-500" />;
      case "failed":
        return <XCircle size={18} className="text-red-500" />;
      case "running":
        return <Loader2 size={18} className="text-primary animate-spin" />;
      default:
        return <Loader2 size={18} className="text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      completed: "bg-green-500/10 text-green-500 border-green-500/20",
      failed: "bg-red-500/10 text-red-500 border-red-500/20",
      running: "bg-primary/10 text-primary border-primary/20",
    };

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${styles[status] || styles.running}`}>
        {status.toUpperCase()}
      </span>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={32} className="text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground" data-testid="campaign-history-title">Campaign History</h1>
          <p className="text-muted-foreground mt-1">View and manage past campaigns</p>
        </div>
        <div className="text-sm text-muted-foreground">
          Total: <span className="font-semibold text-foreground">{campaigns.length}</span> campaigns
        </div>
      </div>

      {campaigns.length === 0 ? (
        <div className="bg-card rounded-lg border border-border p-12 text-center shadow-sm">
          <p className="text-muted-foreground">No campaigns found. Start your first campaign!</p>
        </div>
      ) : (
        <div className="bg-card rounded-lg border border-border shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full" data-testid="campaign-history-table">
              <thead className="bg-accent/50 border-b border-border">
                <tr>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Campaign ID</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Objective</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Status</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-foreground">Created At</th>
                  <th className="text-right px-6 py-4 text-sm font-semibold text-foreground">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {campaigns.map((campaign, idx) => (
                  <tr 
                    key={campaign.campaign_id} 
                    className="hover:bg-accent/30 transition-colors cursor-pointer"
                    onClick={() => navigate(`/campaign/${campaign.campaign_id}`)}
                    data-testid={`campaign-row-${idx}`}
                  >
                    <td className="px-6 py-4">
                      <span className="text-sm font-mono text-foreground">{campaign.campaign_id.slice(0, 8)}...</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-sm text-foreground line-clamp-1">{campaign.objective}</span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(campaign.status)}
                        {getStatusBadge(campaign.status)}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-sm text-muted-foreground">
                        {new Date(campaign.created_at).toLocaleString()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/campaign/${campaign.campaign_id}`);
                        }}
                        data-testid={`view-campaign-${idx}`}
                        className="inline-flex items-center space-x-1 text-primary hover:text-primary/80 transition-colors"
                      >
                        <span className="text-sm font-medium">View</span>
                        <ChevronRight size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default CampaignHistory;
