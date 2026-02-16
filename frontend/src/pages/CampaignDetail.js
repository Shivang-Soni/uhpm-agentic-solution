import React from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import { ArrowLeft, CheckCircle2, XCircle, Loader2, Clock } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = BACKEND_URL;

const CampaignDetail = () => {
  const { campaignId } = useParams();
  const navigate = useNavigate();
  const [campaign, setCampaign] = React.useState(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    fetchCampaign();
  }, [campaignId]);

  const fetchCampaign = async () => {
    try {
      const response = await axios.get(`${API}/campaigns/${campaignId}`);
      setCampaign(response.data);
    } catch (error) {
      console.error("Failed to fetch campaign:", error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 size={20} className="text-green-500" />;
      case "failed":
        return <XCircle size={20} className="text-red-500" />;
      case "running":
        return <Loader2 size={20} className="text-primary animate-spin" />;
      default:
        return <Clock size={20} className="text-muted-foreground" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={32} className="text-primary animate-spin" />
      </div>
    );
  }

  if (!campaign || campaign.error) {
    return (
      <div className="space-y-6">
        <button
          onClick={() => navigate("/history")}
          className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft size={20} />
          <span>Back to History</span>
        </button>
        <div className="bg-card rounded-lg border border-border p-12 text-center">
          <p className="text-muted-foreground">Campaign not found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <button
        onClick={() => navigate("/history")}
        data-testid="back-to-history-button"
        className="flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft size={20} />
        <span>Back to History</span>
      </button>

      <div>
        <h1 className="text-3xl font-bold text-foreground" data-testid="campaign-detail-title">Campaign Details</h1>
        <p className="text-muted-foreground mt-1">{campaign.campaign_id}</p>
      </div>

      <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-foreground mb-4">Objective</h2>
        <p className="text-foreground" data-testid="campaign-objective">{campaign.objective}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-foreground mb-6">Execution Timeline</h2>
            
            <div className="relative" data-testid="detail-timeline">
              <div className="absolute left-3 top-0 bottom-0 w-0.5 bg-border"></div>
              
              <div className="space-y-6">
                {(campaign.history || []).map((step, idx) => (
                  <div key={idx} className="relative flex items-start space-x-4">
                    <div className="relative z-10 flex items-center justify-center w-6 h-6 rounded-full bg-card border-2 border-border">
                      {getStatusIcon(step.status)}
                    </div>
                    
                    <div className="flex-1 pb-6">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-foreground">{step.step}</h3>
                        <span className="text-xs text-muted-foreground">
                          {new Date(step.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Status: {step.status} • Retries: {step.retry_count}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-foreground mb-4">Campaign Info</h2>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-muted-foreground">Status:</span>
                <span className="ml-2 text-foreground font-medium" data-testid="detail-status">{campaign.status}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Current Action:</span>
                <span className="ml-2 text-foreground font-medium">{campaign.current_action}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Created:</span>
                <span className="ml-2 text-foreground">{new Date(campaign.created_at).toLocaleString()}</span>
              </div>
            </div>
          </div>

          {campaign.vector_memory && campaign.vector_memory.documents.length > 0 && (
            <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-foreground mb-4">Vector Memory</h2>
              <div className="space-y-3">
                {campaign.vector_memory.documents.map((doc, idx) => (
                  <div key={idx} className="p-3 bg-accent/50 rounded-lg border border-border">
                    <p className="text-xs text-foreground">{doc}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Distance: {campaign.vector_memory.distances[idx].toFixed(3)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CampaignDetail;
